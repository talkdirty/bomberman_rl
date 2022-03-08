import numpy as np

import settings as s
import settings
import events as e

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

def is_subset(set, subset):
    return all(x in set for x in subset)

def is_suicide(events):
    return is_subset(events, [e.BOMB_EXPLODED, e.KILLED_SELF, e.GOT_KILLED])

def is_good_bomb_placement(events):
    num_crates = events.count(e.CRATE_DESTROYED)
    if num_crates > 0 and is_subset(events, [e.BOMB_EXPLODED]) and not is_suicide(events):
        return num_crates

BOMB_FLED = "BOMB_FLED"
WALKS_INTO_BOMB_RADIUS = "WALKS_INTO_BOMB_RADIUS"
AGENT_MOVED_OUT_OF_BOMB_TILE = "AGENT_MOVED_OUT_OF_BOMB_TILE"

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
        path, _ = finder.find_path(g_agent, g_bomb, g)
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

def feat_on_bomb(state):
    return int(agent_on_bomb(state))


def bomb_pathfinding_grid(state):
    grid = np.zeros_like(state['field'])
    grid[state['field'] == 1] = -1 # Crates are unwalkable
    grid[state['field'] == -1] = -1 # Walls are unwalkable
    grid[state['field'] == 0] = 1 # Floor is walkable
    g = Grid(matrix=grid)
    return g

# TODO refactor
def agent_on_bomb(state):
    if state is None or state['field'] is None:
        return
    g = bomb_pathfinding_grid(state)
    g_agent = g.node(state['self'][3][1], state['self'][3][0])
    for bomb in state['bombs']:
        g_bomb = g.node(bomb[0][1], bomb[0][0])
        if g_agent.x == g_bomb.x and g_agent.y == g_bomb.y:
            # Agent and bomb are still on same tile
            return True
    return False

def bomb_in_line_of_sight(state, bomb_range=s.BOMB_POWER):
    """"
    Pathfinding scenario: Our agent, all bombs, all obstacles.
    """
    if state is None or state['field'] is None:
        return
    g = bomb_pathfinding_grid(state)
    g_agent = g.node(state['self'][3][1], state['self'][3][0])
    for bomb in state['bombs']:
        g_bomb = g.node(bomb[0][1], bomb[0][0])
        if g_agent.x == g_bomb.x and g_agent.y == g_bomb.y:
            # Agent and bomb are still on same tile
            return True
        path, _ = finder.find_path(g_agent, g_bomb, g)
        if len(path) > 1 and len(path) <= bomb_range + 1:
            path = np.array(path)
            x_line_of_sight = np.all(path[:, 0] == path[0, 0])
            y_line_of_sight = np.all(path[:, 1] == path[0, 1])
            if x_line_of_sight or y_line_of_sight:
                return True

def bomb_fled_event(old_state, new_state, action):
    """Produce Event if agent took step towards evading a bomb."""
    if not old_state:
        # First step: this event can't be
        return
    if bomb_in_line_of_sight(old_state) and not bomb_in_line_of_sight(new_state):
        return BOMB_FLED
    if not bomb_in_line_of_sight(old_state) and bomb_in_line_of_sight(new_state):
        return WALKS_INTO_BOMB_RADIUS

def agent_moved_out_of_bomb_tile(old_state, new_state, action):
    if not old_state:
        return
    if agent_on_bomb(old_state) and not agent_on_bomb(new_state):
        return AGENT_MOVED_OUT_OF_BOMB_TILE

def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    if is_suicide(events):
        return -3
    crates_destroyed = is_good_bomb_placement(events)
    if crates_destroyed is not None:
        return crates_destroyed * 2.5
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.INVALID_ACTION: -1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 1,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.3,
        BOMB_FLED: 3,
        WALKS_INTO_BOMB_RADIUS: -3,
        AGENT_MOVED_OUT_OF_BOMB_TILE: 2
    }
    # BOMB_DROPPED and walks into bomb radius cancel each other
    if is_subset(events, [e.BOMB_DROPPED, WALKS_INTO_BOMB_RADIUS]):
        events.remove(WALKS_INTO_BOMB_RADIUS)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum

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
    gym_f[field == 1] = 1
    gym_f[field == -1] = 2
    for other in others:
        gym_f[other[3][1], other[3][0]] = 3
    for bomb in bombs:
        gym_f[bomb[0][1], bomb[0][0]] = 6
    gym_f[self[3][1], self[3][0]] = 4
    c = gym_coins(coins)
    gym_f[c == 1] = 5

    gym_f[(explosion_map != 0).T] = 7
    
    return gym_f

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
    return feature

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
        'field': gym_field(game_state['field'], game_state['others'], game_state['self'], game_state['coins'], game_state['bombs'], game_state['explosion_map']),
        "bomb_awareness": feat_bomb_situational_awareness(game_state),
        "bomb_on": feat_on_bomb(game_state)
        #'bombs': gym_bombs(game_state['bombs']),
        #'explosions': gym_explosions(game_state['explosion_map']),
        #'coins': gym_coins(game_state['coins']),
        #'other_bombs': gym_other_bombs(game_state['others'])
    }