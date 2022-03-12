import numpy as np

import numba

import bombergym.environments.plain.features as plain_features
from bombergym.environments.plain.navigation import Tile
from bombergym.environments.reduced.features import get_is_on_bomb, get_walls

def pathfinding_distance(grid, coord, tile):
    x, y = coord
    target_coords = np.transpose((grid == tile).nonzero())
    if len(target_coords) == 0:
        return np.nan
    min_dist = np.min(np.abs(target_coords[:, 0] - y) + np.abs(target_coords[:, 1] - x)) + 1
    return min_dist

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

@numba.jit(forceobj=True)
def state_to_gym(game_state):
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    awareness_features = get_awareness_features(game_state)
    is_on_bomb = get_is_on_bomb(game_state)
    walls = get_walls(game_state)
    features =  np.hstack((awareness_features, walls, is_on_bomb))
    return features