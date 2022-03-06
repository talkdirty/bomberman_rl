import numpy as np

import settings

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

def gym_field(field: np.ndarray) -> np.array:
    gym_f = np.zeros_like(field, dtype=np.int64)
    gym_f[field == 1] = 1
    gym_f[field == -1] = 2
    return gym_f.flatten()

def gym_bombs(bomb_state: list) -> np.array:
    gym_b = np.zeros((settings.ROWS, settings.COLS), dtype=np.int64)
    for bomb in bomb_state:
        gym_b[bomb[0][1], bomb[0][0]] = bomb[1] + 1 # bomb[1]==0: about to explode
    return gym_b.flatten()

def gym_explosions(explosion_state: np.array) -> np.array:
    return explosion_state.flatten()

def gym_coins(coin_state: list) -> np.ndarray:
    feature = np.zeros((settings.ROWS, settings.COLS), dtype=np.int64)
    for coin in coin_state:
        feature[coin[0], coin[1]] = 1
    return feature.flatten()

def gym_other_bombs(others: list) -> np.ndarray:
    feature = np.zeros(3, dtype=np.int64)
    for i, other in enumerate(others):
        feature[i] = 1 if other[2] else 0
    return feature

def gym_others(others: list) -> np.ndarray:
    feature = np.zeros((settings.ROWS, settings.COLS), dtype=np.int64)
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
    return {
        'field': gym_field(game_state['field']),
        'bombs': gym_bombs(game_state['bombs']),
        'explosions': gym_explosions(game_state['explosion_map']),
        'coins': gym_coins(game_state['coins']),
        'other_bombs': gym_other_bombs(game_state['others']),
        'others': gym_others(game_state['others'])
    }