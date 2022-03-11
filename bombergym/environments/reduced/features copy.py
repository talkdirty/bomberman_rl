import numpy as np

import bombergym.settings as s
import bombergym.environments.plain.features as plain_features

def get_danger(game_state: dict):
    grid = plain_features.state_to_gym(game_state)
    own_x, own_y = game_state['self'][3]
    
    distance = 0
    for i in range(own_x, s.COLS):
        grid[own_y, i]
        if distance > s.BOMB_POWER:

            break
        distance += 1

    import ipdb; ipdb.set_trace()

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
    danger = get_danger(game_state)
    if game_state is None:
        return None
    pass