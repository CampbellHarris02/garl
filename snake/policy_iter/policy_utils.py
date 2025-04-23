
## File: policy_utils.py
import random
import math
from collections import defaultdict
from snake.policy_iter.snake_game import SnakeGame
from snake.policy_iter.constants import ACTIONS, STEPS_PER_GAME, FOOD_MULTIPLE, STEPS_MULTIPLE, VISITED_MULTIPLE


def random_policy():
    policy = defaultdict(lambda: random.choice(ACTIONS))
    game = SnakeGame()
    state = game.reset()
    visited_states = set()

    for _ in range(STEPS_PER_GAME):
        visited_states.add(state)
        action = random.choice(ACTIONS)
        policy[state] = action
        state, done = game.step(action)
        if done:
            break

    return policy, visited_states

def evaluate_policy(policy):
    game = SnakeGame()
    state = game.reset()
    visited = set([state])
    food_collected = 0
    for _ in range(STEPS_PER_GAME):
        if state not in policy:
            policy[state] = random.choice(ACTIONS)  # ← grow the policy here

        prev_score = game.score
        action = policy[state]
        state, done = game.step(action)
        visited.add(state)
        if game.score > prev_score:
            food_collected += 1
        if done:
            break
    fitness = STEPS_MULTIPLE * game.steps + FOOD_MULTIPLE * food_collected + VISITED_MULTIPLE * len(visited)
    return fitness, visited

def policy_distance(p1, p2):
    keys = set(p1.keys()).union(set(p2.keys()))
    diff = sum(p1[k] != p2[k] for k in keys)
    return diff / len(keys) if keys else 0.0
