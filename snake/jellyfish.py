# jellyfish.py
import random
from collections import defaultdict


def replicate(parent_policy, action_space, num_children, mutation_rate):
    """
    Generate num_children mutated versions of a single parent policy.
    Each child is a mutation of the parent, with a given action_space.
    """
    children = []
    for _ in range(num_children):
        child = defaultdict(lambda: random.choice(action_space))
        for k in parent_policy:
            if random.random() < mutation_rate:
                child[k] = random.choice(action_space)
            else:
                child[k] = parent_policy[k]
        children.append(child)
    return children
