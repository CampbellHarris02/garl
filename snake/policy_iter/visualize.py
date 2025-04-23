## File: visualize.py
import matplotlib.pyplot as plt # type: ignore
import numpy as np

def plot_fitness(history):
    plt.plot(history)
    plt.title("Best Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("log(steps) + score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_individuals(history_top10):
    hist_top_array = np.array(history_top10)
    for i in range(min(10, hist_top_array.shape[1])):
        plt.plot(hist_top_array[:, i], label=f'Top {i+1}', alpha=0.8)
    plt.title("Top 10 Individual Fitness per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_exploration(explored_states):
    plt.plot(explored_states)
    plt.title("Best Individual's Explored States per Generation")
    plt.xlabel("Generation")
    plt.ylabel("# Unique States Visited")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_statistics(avg_hist, std_hist):
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