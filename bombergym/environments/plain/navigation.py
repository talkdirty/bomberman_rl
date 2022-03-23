from matplotlib.style import available
import numpy as np

from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.finder.a_star import AStarFinder

pathfinder = AStarFinder(diagonal_movement=DiagonalMovement.never)

class Tile:
    EXPLOSION = -4
    BOMB = -3
    WALL = -2
    ENEMY = -1
    EMPTY = 0
    CRATE = 1
    SELF = 2
    COIN = 3

    @staticmethod
    def walkable(tile):
        return tile == Tile.EMPTY or tile == Tile.COIN


def bomb_pathfinding_grid(state):
    grid = np.zeros_like(state['field'])
    grid[state['field'] == 1] = -1 # Crates are unwalkable
    grid[state['field'] == -1] = -1 # Walls are unwalkable
    grid[state['field'] == 0] = 1 # Floor is walkable
    g = Grid(matrix=grid)
    return g

def bomb_pathfinding_grid_neighbor(state):
    grid = np.zeros_like(state['field'])
    grid[state['field'] == 1] = 1 # Crates are unwalkable
    grid[state['field'] == -1] = -1 # Walls are unwalkable
    grid[state['field'] == 0] = 1 # Floor is walkable
    g = Grid(matrix=grid)
    return g

def get_available_actions(gym_state, orig_state):
    available_actions = ['WAIT']
    if orig_state['self'][2]:
        available_actions.append('BOMB')
    agent_x, agent_y = orig_state['self'][3]
    if Tile.walkable(gym_state[agent_y, agent_x - 1]):
        available_actions.append('LEFT')
    if Tile.walkable(gym_state[agent_y, agent_x + 1]):
        available_actions.append('RIGHT')
    if Tile.walkable(gym_state[agent_y + 1, agent_x]):
        available_actions.append('BOTTOM')
    if Tile.walkable(gym_state[agent_y - 1, agent_x]):
        available_actions.append('TOP')
    return available_actions
