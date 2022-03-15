import numpy as np

import bombergym.settings as s

def compute_bomb_blast(field, bomb_coord):
    blast = np.zeros(field.shape, dtype=np.float32)
    b_x, b_y = bomb_coord
    for i in range(s.BOMB_POWER + 1):
        left_x, left_y = b_x - i, b_y
        if left_x < 0 or left_y < 0 or left_x >= field.shape[1] or left_y >= field.shape[0]:
            break
        if field[left_y, left_x] == -1:
            break
        blast[left_y, left_x] = 1.
    for i in range(s.BOMB_POWER + 1):
        right_x, right_y = b_x + i, b_y
        if right_x < 0 or right_y < 0 or right_x >= field.shape[1] or right_y >= field.shape[0]:
            break
        if field[right_y, right_x] == -1:
            break
        blast[right_y, right_x] = 1.
    for i in range(s.BOMB_POWER + 1):
        top_x, top_y = b_x, b_y - i
        if top_x < 0 or top_y < 0 or top_x >= field.shape[1] or top_y >= field.shape[0]:
            break
        if field[top_y, top_x] == -1:
            break
        blast[top_y, top_x] = 1.
    for i in range(s.BOMB_POWER + 1):
        bottom_x, bottom_y = b_x, b_y + i
        if bottom_x < 0 or bottom_y < 0 or bottom_x >= field.shape[1] or bottom_y >= field.shape[0]:
            break
        if field[bottom_y, bottom_x] == -1:
            break
        blast[bottom_y, bottom_x] = 1.
    return blast


def gym_field_danger(field, bombs, explosion_map):
    """
    Represent danger.
    A danger of one represents instadeath (ongoing explosion)
    A danger > 0 represents looming danger (bomb is ticking)
    """
    f = np.zeros(field.shape, dtype=np.float32)
    f[explosion_map != 0] = 1.
    for bomb in bombs:
        blast = compute_bomb_blast(field, bomb[0])
        blast = blast / (bomb[1] + 2)
        f[blast != 0] = blast[blast != 0]
    return f

def gym_field_crate(field):
    f = np.zeros_like(field)
    f[field == 1] = 1.
    return f

def gym_field_walls(field):
    f = np.zeros_like(field)
    f[field == -1] = 1.
    return f

def gym_field_coins(field, coins):
    f = np.zeros_like(field)
    for coin in coins:
        x, y = coin
        f[y, x] = 1.
    return f

def gym_field_agents(field, self, others):
    f = np.zeros_like(field)
    x, y = self[3]
    f[y, x] = 1.
    for other in others:
        x, y = other[3]
        f[y, x] = -1.
    return f

    
def state_to_gym(game_state: dict) -> dict:
    """
    :param game_state:  A dictionary describing the current game board.
    :return: dict compatible with gym observation space
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    crate_field = gym_field_crate(game_state['field'].T)
    danger_field = gym_field_danger(game_state['field'].T, game_state['bombs'], game_state['explosion_map'].T)
    wall_field = gym_field_walls(game_state['field'].T)
    money_field = gym_field_coins(game_state['field'].T, game_state['coins'])
    agent_field = gym_field_agents(game_state['field'].T, game_state['self'],game_state['others'])
    full = np.dstack((crate_field, danger_field, wall_field, money_field, agent_field))
    return full.swapaxes(0,2)
