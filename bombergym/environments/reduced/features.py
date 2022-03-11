import numpy as np

import bombergym.settings as s
import bombergym.environments.plain.features as plain_features
from bombergym.environments.plain.navigation import Tile


def get_x_distance_if_reachable(grid, coords, tile, direction):
    """Get distance to a given Tile, unless Wall is in the way. X direction"""
    x, y = coords
    reachable = False
    distance = None
    coord = x 
    while True:
        if coord >= s.COLS or coord < 0:
            break
        if grid[y, coord] == tile:
            distance = abs(coord - x)
            if distance == 0:
                distance = np.nan
            reachable = True
        if grid[y, coord] == Tile.WALL and not reachable:
            break
        coord += direction
    if reachable:
        return distance

def get_y_distance_if_reachable(grid, coords, tile, direction):
    """Get distance to a given Tile, unless Wall is in the way. X direction"""
    x, y = coords
    reachable = False
    distance = None
    coord = y 
    while True:
        if coord >= s.ROWS or coord < 0:
            break
        if grid[coord, x] == tile:
            distance = abs(coord - y)
            if distance == 0:
                distance = np.nan
            reachable = True
        if grid[coord, x] == Tile.WALL and not reachable:
            break
        coord += direction
    if reachable:
        return distance

def get_enemy_distances(game_state: dict):
    MAX_PLAYERS = 4
    x, y = game_state['self'][3]
    distances = np.array([np.nan] * MAX_PLAYERS)
    others = game_state['others']
    for i in range(len(others)):
        x_e, y_e = others[i][3]
        distances[i] = np.sqrt((x - x_e) ** 2 + (y - y_e) ** 2)
    return np.nan_to_num(1 / distances)

def get_walls(game_state: dict):
    grid = plain_features.state_to_gym(game_state)
    x, y = game_state['self'][3]
    wall_r = 1. if grid[y, x+1] == Tile.WALL else 0.
    wall_l = 1. if grid[y, x-1] == Tile.WALL else 0.
    wall_t = 1. if grid[y-1, x] == Tile.WALL else 0.
    wall_b = 1. if grid[y+1, x] == Tile.WALL else 0.
    return np.array([wall_r, wall_l, wall_t, wall_b])

def get_objects(game_state: dict):
    """Returns object reachable by the agent, meaning no wall in middle"""
    # TODO: try dividing distance by bomb timer: np.abs(s.BOMB_TIMER)
    grid = plain_features.state_to_gym(game_state)
    coords = game_state['self'][3]
    bomb_r = get_x_distance_if_reachable(grid, coords, Tile.BOMB, +1)
    bomb_l = get_x_distance_if_reachable(grid, coords, Tile.BOMB, -1)
    bomb_t = get_y_distance_if_reachable(grid, coords, Tile.BOMB, +1)
    bomb_b = get_y_distance_if_reachable(grid, coords, Tile.BOMB, -1)

    explosion_r = get_x_distance_if_reachable(grid, coords, Tile.EXPLOSION, +1)
    explosion_l = get_x_distance_if_reachable(grid, coords, Tile.EXPLOSION, -1)
    explosion_t = get_y_distance_if_reachable(grid, coords, Tile.EXPLOSION, +1)
    explosion_b = get_y_distance_if_reachable(grid, coords, Tile.EXPLOSION, -1)

    crate_r = get_x_distance_if_reachable(grid, coords, Tile.CRATE, +1)
    crate_l = get_x_distance_if_reachable(grid, coords, Tile.CRATE, -1)
    crate_t = get_y_distance_if_reachable(grid, coords, Tile.CRATE, +1)
    crate_b = get_y_distance_if_reachable(grid, coords, Tile.CRATE, -1)

    coin_r = get_x_distance_if_reachable(grid, coords, Tile.COIN, +1)
    coin_l = get_x_distance_if_reachable(grid, coords, Tile.COIN, -1)
    coin_t = get_y_distance_if_reachable(grid, coords, Tile.COIN, +1)
    coin_b = get_y_distance_if_reachable(grid, coords, Tile.COIN, -1)


    return np.nan_to_num(1/np.array(
        [bomb_l, bomb_r, bomb_t, bomb_b, 
        explosion_l, explosion_r, explosion_t, explosion_b,
        crate_l, crate_r, crate_t, crate_b,
        coin_l, coin_r, coin_t, coin_b,
        ], dtype=np.float32
    ))

def get_is_on_bomb(game_state: dict):
    """Return 1. if agent currently on bomb, 0. otherwise"""
    agent_x, agent_y = game_state['self'][3]
    for bomb in game_state['bombs']:
        if bomb[0][0] == agent_x and bomb[0][1] == agent_y:
            return np.array([1.])
    return np.array([0.])
    

def state_to_gym(game_state: dict) -> dict:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: dict compatible with gym observation space

    * Observation space is a low dimensional vector
      1. Danger (r, l, t, b), high if close to bomb or explosion
      2. Coin (r, l, t, b), high if coin is nearby
      3. Crate (r, l, t, b), high if close to a crate
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    objects = get_objects(game_state)
    walls = get_walls(game_state)
    is_on_bomb = get_is_on_bomb(game_state)
    enemy_distances = get_enemy_distances(game_state)
    features =  np.hstack((objects, enemy_distances, walls, is_on_bomb))
    return features