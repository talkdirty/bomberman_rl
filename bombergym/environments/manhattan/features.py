import numpy as np

from pathfinding.core.grid import Grid

import bombergym.environments.plain.features as plain_features
from bombergym.environments.plain.navigation import Tile, pathfinder
from bombergym.environments.reduced.features import get_is_on_bomb, get_walls

def pathfinding_grid(grid):
    g = np.ones_like(grid)
    g[grid == Tile.BOMB] = -1 
    g[grid == Tile.CRATE] = -1 
    g[grid == Tile.WALL] = -1 
    g[grid == Tile.SELF] = -1
    return Grid(matrix=g)

def pathfinding_distance(grid, own_coords, tile):
    x, y = own_coords
    g = pathfinding_grid(grid)
    if grid[y, x] == tile:
        return 1
    if not g.walkable(x, y):
        return np.nan
    agent = g.node(x, y)
    all_tile_coords = np.transpose((grid == tile).nonzero())
    distance_min = np.inf
    # shortest_path = None
    # shortest_thing = None
    for coord in all_tile_coords:
        g.cleanup()
        x_thing, y_thing = coord
        thing = g.node(y_thing, x_thing)
        path, _ = pathfinder.find_path(agent, thing, g)
        if len(path) < distance_min and not len(path) == 0:
            # shortest_path = path
            # shortest_thing = thing
            distance_min = len(path)
    # Debugging:
    # print(g.grid_str(path=shortest_path, start=agent, end=shortest_thing))
    return distance_min

def get_awareness(grid, own_coords, tile):
    x, y = own_coords
    return np.nan_to_num(1/np.array([
        pathfinding_distance(grid, (x - 1, y), tile),
        pathfinding_distance(grid, (x + 1, y), tile),
        pathfinding_distance(grid, (x, y - 1), tile),
        pathfinding_distance(grid, (x, y + 1), tile)
    ]))


def get_awareness_features(game_state: dict):
    grid = plain_features.state_to_gym(game_state)
    coords = game_state['self'][3]

    awareness_bomb = get_awareness(grid, coords, Tile.BOMB)
    awareness_expl = get_awareness(grid, coords, Tile.EXPLOSION)
    awareness_crate = get_awareness(grid, coords, Tile.CRATE)
    awareness_coin = get_awareness(grid, coords, Tile.COIN)
    awareness_enemy = get_awareness(grid, coords, Tile.ENEMY)

    return np.hstack((
        awareness_bomb,
        awareness_expl,
        awareness_crate,
        awareness_coin,
        awareness_enemy
    ))

def state_to_gym(game_state: dict) -> dict:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    awareness_features = get_awareness_features(game_state)
    is_on_bomb = get_is_on_bomb(game_state)
    walls = get_walls(game_state)
    features =  np.hstack((awareness_features, walls, is_on_bomb))
    return features