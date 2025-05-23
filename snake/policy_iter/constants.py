## File: constants.py
GRID_SIZE = 8
SCREEN_SIZE = 600
TILE_SIZE = SCREEN_SIZE // GRID_SIZE
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTIONS = [-1, 0, 1]

STEPS_PER_GAME = 10000
POP_SIZE = 1000
GENS = 20
MUTATION_RATE = 0.0914
FOOD_MULTIPLE = 8.76
STEPS_MULTIPLE = 0.0459
VISITED_MULTIPLE = 0.0472