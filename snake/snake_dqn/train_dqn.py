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

def simulate_trained_model(model, config, stack_size):
    """Simulate the Snake game using a trained DQN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    game = SnakeGame(stack_size=stack_size)
    
    state = game.reset()  # <- Add this to initialize state
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


def compute_reward(score, prev_score, reward_food, reward_step):
    return reward_food if score > prev_score else reward_step


def select_action(state, model, epsilon, device):
    if random.random() < epsilon:
        return random.choice([-1, 0, 1])
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item() - 1  # shift to [-1, 0, 1]


def train_dqn(config, episodes=1000, max_steps=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory_frames = 4
    game = SnakeGame(stack_size=memory_frames)
    state_size = len(game.reset())
    action_size = 3  # Actions: [-1, 0, 1]

    model = DQN(state_size, action_size, config).to(device)
    target_model = DQN(state_size, action_size, config).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()
    memory = deque(maxlen=config["replay_buffer_size"])
    batch_size = int(config["batch_size"])
    gamma = config["gamma"]

    epsilon = config["epsilon_start"]
    epsilon_min = config["epsilon_min"]
    epsilon_decay = config["epsilon_decay"]
    reward_food = config["reward_food"]
    reward_step = config["reward_step"]

    total_rewards, epsilon_history, loss_history = [], [], []

    for episode in range(episodes):
        game = SnakeGame(stack_size=memory_frames)
        game.snake = deque([(5, 5 - i) for i in range(random.randint(1, 30))])
        state = np.array(game.reset(), dtype=np.float32)
        total_reward = 0
        episode_loss = 0
        prev_score = 0

        for _ in range(max_steps):
            action = select_action(state, model, epsilon, device)
            next_state, done = game.step(action)
            next_state = np.array(next_state, dtype=np.float32)

            reward = compute_reward(game.score, prev_score, reward_food, reward_step)
            prev_score = game.score
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                actions_tensor = torch.tensor([a+1 for a in actions], dtype=torch.long).unsqueeze(1).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = model(states_tensor).gather(1, actions_tensor).squeeze()
                next_q_values = target_model(next_states_tensor).max(1)[0]
                target_q = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                loss = loss_fn(q_values, target_q.detach())
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
        
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "stack_size": len(game.state_stack) if hasattr(game, "state_stack") else 1,
    }, "trained_dqn.pth")

    return np.mean(total_rewards[-10:]), total_rewards, epsilon_history, loss_history, model



if __name__ == "__main__":
    avg, total_rewards, epsilon_history, loss_history, model = train_dqn(default_config)
    plot_rewards(total_rewards)
    plot_epsilon(epsilon_history)
    plot_loss(loss_history)

    # Load model and config
    checkpoint = torch.load("trained_dqn.pth")
    config = checkpoint["config"]
    stack_size = checkpoint["stack_size"]
    state_size = 9 * stack_size

    # Rebuild the model with correct input size
    model = DQN(state_size, 3, config)
    model.load_state_dict(checkpoint["model_state_dict"])

    simulate_trained_model(model, config, stack_size=stack_size)
