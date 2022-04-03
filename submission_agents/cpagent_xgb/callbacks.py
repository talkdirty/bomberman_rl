import numpy as np
from . import plain_features
import xgboost as xgb
import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
    :return: array compatible with gym observation space

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

def get_available_actions(orig_state, gym_state):
    available_actions = ['WAIT']
    if orig_state['self'][2]:
        available_actions.append('BOMB')
    agent_x, agent_y = orig_state['self'][3]
    if Tile.walkable(gym_state[agent_y, agent_x - 1]):
        available_actions.append('LEFT')
    if Tile.walkable(gym_state[agent_y, agent_x + 1]):
        available_actions.append('RIGHT')
    if Tile.walkable(gym_state[agent_y + 1, agent_x]):
        available_actions.append('DOWN')
    if Tile.walkable(gym_state[agent_y - 1, agent_x]):
        available_actions.append('UP')
    return available_actions

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.model = xgb.Booster({'nthread': 4})
    self.model.load_model('bst0001.model')

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    obs = state_to_gym(game_state)
    actions_avail = get_available_actions(game_state, plain_features.state_to_gym(game_state))
    actions_avail = [ACTIONS.index(a) for a in actions_avail]
    action_probs = self.model.predict(xgb.DMatrix(obs[None, ...]))
    action_probs_sorted = action_probs.argsort().squeeze()
    action = ACTIONS.index('WAIT')
    for i in range(len(action_probs_sorted)):
        action_candidate = action_probs_sorted[i]
        if action_candidate in actions_avail:
            action = action_candidate
            break
    return ACTIONS[action]
