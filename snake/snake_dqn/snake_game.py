## File: snake_game.py
import random
import numpy as np
from collections import deque
GRID_SIZE = 8
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [-1, 0, 1]

class SnakeGame:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.state_stack = deque(maxlen=stack_size)
        self.reset()

    def reset(self):
        self.snake = deque([(5, 5)])
        self.direction = random.randint(0, 3)
        self.place_food()
        self.score = 0
        self.steps = 0
        self.head_history = deque(maxlen=24)
        self.state_stack = deque(maxlen=self.stack_size)
        state = self.get_state()
        for _ in range(self.stack_size):
            self.state_stack.append(state)
        return self.get_stacked_state()
    
    def place_food(self):
        while True:
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.food not in self.snake:
                break

    def step(self, action):
        self.direction = (self.direction + action) % 4
        dx, dy = DIRS[self.direction]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)

        self.steps += 1
        self.head_history.appendleft(head)

        if len(self.snake) < 3 and len(self.head_history) >= 12:
            recent = list(self.head_history)
            for window in [4, 6]:
                if len(recent) >= 2 * window and recent[:window] == recent[window:2*window]:
                    return self.get_stacked_state(), True

        if head[0] < 0 or head[0] >= GRID_SIZE or head[1] < 0 or head[1] >= GRID_SIZE or head in self.snake:
            return self.get_stacked_state(), True

        self.snake.appendleft(head)
        if head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        state = self.get_state()
        self.state_stack.append(state)
        return self.get_stacked_state(), False

    def get_stacked_state(self):
        return np.concatenate(self.state_stack)

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_dx = int((self.food[0] - head_x) / max(abs(self.food[0] - head_x), 1))
        food_dy = int((self.food[1] - head_y) / max(abs(self.food[1] - head_y), 1))

        def danger(offset):
            dx, dy = DIRS[(self.direction + offset) % 4]
            nx, ny = head_x + dx, head_y + dy
            return int(nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or (nx, ny) in self.snake)

        return np.array([
            food_dx, food_dy,
            self.direction,
            danger(0), danger(1), danger(-1),
            min(len(self.snake), 20),
            head_x, head_y
        ], dtype=np.float32)
