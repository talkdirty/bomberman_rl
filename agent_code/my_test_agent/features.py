import torch
import numpy as np

def flatten_bomb_state(bomb_state: list) -> np.array:
    MAX_BOMBS = 4
    features = np.zeros(MAX_BOMBS * 3)
    for i, bomb in enumerate(bomb_state):
        features[i*3 + 0] = bomb[0][0]
        features[i*3 + 1] = bomb[0][1]
        features[i*3 + 2] = bomb[1]
    return features

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

    
def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    stacked = np.hstack((
        game_state['field'].flatten(),
        flatten_bomb_state(game_state['bombs']),
        agents_to_features(game_state).flatten(),
    )).astype(np.float32)
    return torch.from_numpy(stacked)