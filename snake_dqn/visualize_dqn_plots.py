import matplotlib.pyplot as plt # type: ignore

def plot_rewards(total_rewards):
    plt.figure(figsize=(10, 4))
    plt.plot(total_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward (score + log(steps + 1))")
    plt.title("Training Reward Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_epsilon(epsilon_values):
    plt.figure(figsize=(10, 4))
    plt.plot(epsilon_values, label="Epsilon", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_loss(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Average Loss", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
