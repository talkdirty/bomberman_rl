import numpy as np

from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder


pathfinder = AStarFinder(diagonal_movement=DiagonalMovement.never)

def bomb_pathfinding_grid(state):
    grid = np.zeros_like(state['field'])
    grid[state['field'] == 1] = -1 # Crates are unwalkable
    grid[state['field'] == -1] = -1 # Walls are unwalkable
    grid[state['field'] == 0] = 1 # Floor is walkable
    g = Grid(matrix=grid)
    return g
