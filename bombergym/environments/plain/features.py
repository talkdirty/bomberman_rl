import numpy as np

import bombergym.settings as s

from .navigation import bomb_pathfinding_grid, pathfinder, Tile

# TODO lots of unused features and old code
# TODO refactor
def feat_bomb_situational_awareness(state, bomb_range=s.BOMB_POWER):
    """"
    Pathfinding scenario: Our agent, all bombs, all obstacles.
    """
    if state is None or state['field'] is None:
        return
    g = bomb_pathfinding_grid(state)
    g_agent = g.node(state['self'][3][1], state['self'][3][0])
    dist_x, dist_y = 0, 0
    for bomb in state['bombs']:
        g_bomb = g.node(bomb[0][1], bomb[0][0])
        path, _ = pathfinder.find_path(g_agent, g_bomb, g)
        if len(path) > 1 and len(path) <= bomb_range + 1:
            path = np.array(path)
            x_line_of_sight = np.all(path[:, 0] == path[0, 0])
            y_line_of_sight = np.all(path[:, 1] == path[0, 1])
            if x_line_of_sight:
                # TODO pick path to shortest bomb
                # Negative: to right, Positive: to top
                dist_x = (path[0, :] - path[-1, :])[1]
            if y_line_of_sight:
                # Negative: Upwards, Positive: downwards
                dist_y = (path[0, :] - path[-1, :])[0]
    return np.array([dist_x, dist_y])

#def feat_on_bomb(state):
#    return int(agent_on_bomb(state))

def agent_to_features(agent: tuple) -> np.array:
    return np.array([
        agent[2], # "Bomb available" Bool
        agent[3][0], # Agent x
        agent[3][1], # Agent y
    ])

def agents_to_features(game_state: dict) -> np.array:
    self_feature = agent_to_features(game_state['self'])
    other_features = np.zeros((4, 3))
    for i in range(len(game_state['others'])): # Max. 3 other agents
        other_features[i+1, :] = agent_to_features(game_state['others'][i])
    other_features[0, :] = self_feature
    return other_features

def gym_field(field: np.ndarray, others: list, self: tuple, coins: dict, bombs: list, explosion_map: np.ndarray) -> np.array:
    gym_f = np.zeros_like(field, dtype=np.int64)
    gym_f[(field == 1).T] = Tile.CRATE
    gym_f[(field == -1).T] = Tile.WALL
    for other in others:
        gym_f[other[3][1], other[3][0]] = Tile.ENEMY
    for bomb in bombs:
        gym_f[bomb[0][1], bomb[0][0]] = Tile.BOMB
    gym_f[self[3][1], self[3][0]] = Tile.SELF
    c = gym_coins(coins)
    gym_f[(c == 1).T] = Tile.COIN

    gym_f[(explosion_map != 0).T] = Tile.EXPLOSION
    
    return gym_f

def gym_bombs(bomb_state: list) -> np.array:
    gym_b = np.zeros((s.ROWS, s.COLS), dtype=np.int64)
    for bomb in bomb_state:
        gym_b[bomb[0][1], bomb[0][0]] = bomb[1] + 1 # bomb[1]==0: about to explode
    return gym_b.flatten()

def gym_explosions(explosion_state: np.array) -> np.array:
    return explosion_state.flatten()

def gym_coins(coin_state: list) -> np.ndarray:
    feature = np.zeros((s.ROWS, s.COLS), dtype=np.int64)
    for coin in coin_state:
        feature[coin[0], coin[1]] = 1
    return feature

def gym_other_bombs(others: list) -> np.ndarray:
    feature = np.zeros(3, dtype=np.int64)
    for i, other in enumerate(others):
        feature[i] = 1 if other[2] else 0
    return feature

def gym_others(others: list) -> np.ndarray:
    feature = np.zeros((s.ROWS, s.COLS), dtype=np.int64)
    for other in others:
        feature[other[3][1], other[3][0]] = 1
    return feature.flatten()

    
def state_to_gym(game_state: dict) -> dict:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: dict compatible with gym observation space
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    return gym_field(game_state['field'], game_state['others'], game_state['self'], game_state['coins'], game_state['bombs'], game_state['explosion_map'])


    #return {
    #    'field': gym_field(game_state['field'], game_state['others'], game_state['self'], game_state['coins'], game_state['bombs'], game_state['explosion_map']),
    #    "bomb_awareness": feat_bomb_situational_awareness(game_state),
    #    "bomb_on": feat_on_bomb(game_state)
    #    #'bombs': gym_bombs(game_state['bombs']),
    #    #'explosions': gym_explosions(game_state['explosion_map']),
    #    #'coins': gym_coins(game_state['coins']),
    #    #'other_bombs': gym_other_bombs(game_state['others'])
    #}