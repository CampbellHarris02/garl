import random
import numpy as np
from train_dqn import train_dqn
from rich.console import Console
from rich.progress import track

console = Console()

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
    "hidden_2": (16, 128)
}

POP_SIZE = 20
GENERATIONS = 10

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
    }

def mutate_config(cfg):
    new_cfg = cfg.copy()
    for k in PARAM_BOUNDS:
        if random.random() < 0.3:
            if isinstance(PARAM_BOUNDS[k][0], int):
                new_cfg[k] = random.randint(*PARAM_BOUNDS[k])
            else:
                new_cfg[k] = random.uniform(*PARAM_BOUNDS[k])
    return new_cfg

def crossover_configs(cfg1, cfg2):
    return {k: random.choice([cfg1[k], cfg2[k]]) for k in cfg1}

def evaluate_config(cfg):
    avg_reward, *_ = train_dqn(cfg, episodes=100, max_steps=1000)
    return avg_reward

def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

def garl_dqn():
    population = [random_config() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        console.rule(f"[bold cyan]Meta Generation {gen+1}/{GENERATIONS}")
        scores = []

        for cfg in track(population, description="Evaluating configs"):
            score = evaluate_config(cfg)
            scores.append(score)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        best_idx = np.argmax(scores)

        console.log(f":trophy: [bold green]Best Score: {scores[best_idx]:.4f} with config:")
        console.log(f":bar_chart: [bold cyan]Average Score: {avg_score:.4f}")
        console.log(f":chart_with_upwards_trend: [bold magenta]Std Dev Score: {std_score:.4f}")
        for k, v in population[best_idx].items():
            console.print(f"  [bold]{k}[/bold]: {v}")
        

        # Evolution
        top_half = [cfg for _, cfg in sorted(zip(scores, population), reverse=True)[:POP_SIZE//2]]
        norm_scores = normalize_scores(scores)

        new_pop = [population[best_idx]]  # Elitism
        while len(new_pop) < POP_SIZE:
            parents = random.choices(top_half, k=2)
            child = mutate_config(crossover_configs(*parents))
            new_pop.append(child)

        population = new_pop

    return population[0]

if __name__ == "__main__":
    best = garl_dqn()
    console.rule("[bold green]Best Config Found")
    for k, v in best.items():
        console.print(f'  "{k}": {v},')

