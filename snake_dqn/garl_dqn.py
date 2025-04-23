import random
import numpy as np
from train_dqn import train_dqn
from rich.console import Console
from rich.progress import track

console = Console()

# ── Hyperparameter Bounds ──
PARAM_BOUNDS = {
    "lr": (1e-5, 1e-2),
    "gamma": (0.8, 0.99),
    "epsilon_start": (0.5, 1.0),
    "epsilon_decay": (0.90, 0.999),
    "epsilon_min": (0.001, 0.1),
    "batch_size": (32, 128),
    "replay_buffer_size": (1000, 10000),
    "target_update_interval": (1, 20),
    "hidden_1": (32, 256),
    "hidden_2": (16, 128),
    "reward_food": (0.1, 10),
    "reward_step": (-0.1, 0.1)
}

# ── GA Parameters ──
POP_SIZE = 30
GENERATIONS = 10
MUTATION_RATE = 0.1
ELITE_PERCENT = 0.2
EPISODES = 250
MAX_STEPS = 1000

def random_config():
    return {
        "lr": 10**random.uniform(np.log10(PARAM_BOUNDS["lr"][0]), np.log10(PARAM_BOUNDS["lr"][1])),
        "gamma": random.uniform(*PARAM_BOUNDS["gamma"]),
        "epsilon_start": random.uniform(*PARAM_BOUNDS["epsilon_start"]),
        "epsilon_decay": random.uniform(*PARAM_BOUNDS["epsilon_decay"]),
        "epsilon_min": random.uniform(*PARAM_BOUNDS["epsilon_min"]),
        "batch_size": random.randint(*PARAM_BOUNDS["batch_size"]),
        "replay_buffer_size": int(random.uniform(*PARAM_BOUNDS["replay_buffer_size"])),
        "target_update_interval": int(random.uniform(*PARAM_BOUNDS["target_update_interval"])),
        "hidden_1": random.randint(*PARAM_BOUNDS["hidden_1"]),
        "hidden_2": random.randint(*PARAM_BOUNDS["hidden_2"]),
        "reward_food": random.uniform(*PARAM_BOUNDS["reward_food"]),
        "reward_step": random.uniform(*PARAM_BOUNDS["reward_step"])
    }

def mutate_config(cfg):
    new_cfg = cfg.copy()
    for k in PARAM_BOUNDS:
        if random.random() < MUTATION_RATE:
            if isinstance(PARAM_BOUNDS[k][0], int):
                new_cfg[k] = random.randint(*PARAM_BOUNDS[k])
            else:
                new_cfg[k] = random.uniform(*PARAM_BOUNDS[k])
    return new_cfg

def crossover_configs(cfg1, cfg2):
    return {k: random.choice([cfg1[k], cfg2[k]]) for k in cfg1}

def test_dqn(model, cfg, episodes=5, max_steps=1000):
    from snake_game import SnakeGame
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    lengths = []

    for _ in range(episodes):
        env = SnakeGame()
        state = np.array(env.reset(), dtype=np.float32)

        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item() - 1  # Shift to [-1, 0, 1]

            state, done = env.step(action)
            state = np.array(state, dtype=np.float32)

            if done:
                break

        lengths.append(len(env.snake))

    return np.mean(lengths)


def evaluate_config(cfg):
    import torch.nn as nn
    import torch
    from snake_game import SnakeGame  # Needed for state size and model instantiation
    from model import DQN             # Your DQN model

    avg_reward, _, _, _, model = train_dqn(cfg, episodes=EPISODES, max_steps=MAX_STEPS)
    cfg["__model__"] = model
    return test_dqn(model, cfg)



def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

def garl_dqn():
    population = [random_config() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        console.rule(f"[bold cyan]Meta Generation {gen+1}/{GENERATIONS}")
        scores = [evaluate_config(cfg) for cfg in track(population, description="Evaluating configs")]

        best_score = np.max(scores)
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        best_idx = np.argmax(scores)
        best_cfg = population[best_idx]

        console.log(f":trophy: [bold green]Best Snake Length: {best_score:.2f}")
        console.log(f":bar_chart: [bold cyan]Avg Length (Pop): {avg_score:.2f}")
        console.log(f":chart_with_upwards_trend: [bold magenta]Std Dev: {std_score:.2f}")
        for k, v in best_cfg.items():
            if k != "__model__":
                console.print(f"  [bold]{k}[/bold]: {v}")

        elite_count = int(POP_SIZE * ELITE_PERCENT)
        top_elite = [cfg for score, cfg in sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[:elite_count]]

        new_pop = [best_cfg]
        while len(new_pop) < POP_SIZE:
            parents = random.choices(top_elite, k=2)
            child = mutate_config(crossover_configs(*parents))
            new_pop.append(child)

        population = new_pop

    return best_cfg

if __name__ == "__main__":
    best = {k: v for k, v in garl_dqn().items() if k != "__model__"}
    console.rule("[bold green]Best Config Found")
    for k, v in best.items():
        console.print(f'  "{k}": {v},')
