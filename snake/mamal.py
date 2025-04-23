# mamal.py
import random
from collections import defaultdict

MUTATION_RATE = 0.1

def mutate(policy, action_space):
    new_policy = defaultdict(lambda: random.choice(action_space))
    for k in policy:
        if random.random() < MUTATION_RATE:
            new_policy[k] = random.choice(action_space)
        else:
            new_policy[k] = policy[k]
    return new_policy

def crossover(p1, p2, action_space):
    child = defaultdict(lambda: random.choice(action_space))
    for k in set(p1.keys()).union(p2.keys()):
        child[k] = p1[k] if random.random() < 0.5 else p2[k]
    return child
