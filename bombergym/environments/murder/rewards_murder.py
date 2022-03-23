import numpy as np

import bombergym.original.events as e
import bombergym.settings as s

from bombergym.environments.plain.navigation import bomb_pathfinding_grid, pathfinder, bomb_pathfinding_grid_neighbor


BOMB_FLED = "BOMB_FLED"
WALKS_INTO_BOMB_RADIUS = "WALKS_INTO_BOMB_RADIUS"
AGENT_MOVED_OUT_OF_BOMB_TILE = "AGENT_MOVED_OUT_OF_BOMB_TILE"


MOVED_TOWARDS_OTHER_AGENT = "MOVED_TOWARDS_OTHER_AGENT"
MOVED_TOWARDS_CLOSEST_AGENT = "MOVED_TOWARDS_CLOSEST_AGENT"
IS_NEXT_TO_OTHER_AGENT = "IS_NEXT_TO_OTHER_AGENT"
DROPPED_BOMB_NEXT_TO_OTHER_AGENT = "DROPPED_BOMB_NEXT_TO_OTHER_AGENT"
UNNECESSARY_BOMB = "UNNECESSARY_BOMB"


def is_subset(set, subset):
    return all(x in set for x in subset)

def is_suicide(events):
    return is_subset(events, [e.BOMB_EXPLODED, e.KILLED_SELF, e.GOT_KILLED])

def is_good_bomb_placement(events):
    num_crates = events.count(e.CRATE_DESTROYED)
    if num_crates > 0 and is_subset(events, [e.BOMB_EXPLODED]) and not is_suicide(events):
        return num_crates

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
        path, _ = pathfinder.find_path(g_agent, g_bomb, g)
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

def agent_moved_out_of_bomb_tile(old_state, new_state, action):
    if not old_state:
        return
    if agent_on_bomb(old_state) and not agent_on_bomb(new_state):
        return AGENT_MOVED_OUT_OF_BOMB_TILE

def next_to_other_agent(state):
    """"
    Pathfinding scenario: Our agent, all bombs, all obstacles.
    """
    if state is None or state['field'] is None:
        return
    g = bomb_pathfinding_grid(state)
    g_agent = g.node(state['self'][3][1], state['self'][3][0])
    for agent in state['others']:
        g_other = g.node(agent[3][1], agent[3][0])
        if g_agent.x == g_other.x+1 and g_agent.y == g_other.y:
            return IS_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x-1 and g_agent.y == g_other.y:
            return IS_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x and g_agent.y == g_other.y-1:
            return IS_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x and g_agent.y == g_other.y+1:
            return IS_NEXT_TO_OTHER_AGENT
        

def moved_towards_agent(old_state, state):
    if state is None or state['field'] is None:
        return
    if old_state is None or old_state['field'] is None:
        return
    g = bomb_pathfinding_grid_neighbor(state)
    g_old = bomb_pathfinding_grid_neighbor(old_state)
    g_agent_new = g.node(state['self'][3][1], state['self'][3][0])
    g_agent_old = g.node(old_state['self'][3][1], old_state['self'][3][0])

    agent_dist_new = np.zeros(len(state['others']))
    agent_dist_old = np.zeros(len(old_state['others']))
    for id,agent  in enumerate(old_state['others']):
        if state['others'][id]:
            g_other_old = g.node(agent[3][1], agent[3][0])
            g_other_new = g.node(state['others'][id][3][1], state['others'][id][3][0])
            path_old, _ = pathfinder.find_path(g_agent_old, g_other_old, g_old)
            path_new, _ = pathfinder.find_path(g_agent_new, g_other_new, g)
            agent_dist_new[id] = len(path_new)
            agent_dist_old[id] = len(path_old)
    
    if np.min(agent_dist_new) < np.min(agent_dist_old) and np.min(agent_dist_new) < 7:
        return MOVED_TOWARDS_CLOSEST_AGENT


def dropped_bomb_next_to_other(state):
    """"
    Pathfinding scenario: Our agent, all bombs, all obstacles.
    """
    if state is None or state['field'] is None:
        return
    g = bomb_pathfinding_grid(state)
    g_agent = g.node(state['self'][3][1], state['self'][3][0])
    for agent in state['others']:
        g_other = g.node(agent[3][1], agent[3][0])
        if g_agent.x == g_other.x+1 and g_agent.y == g_other.y:
            if e.BOMB_DROPPED:
                return DROPPED_BOMB_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x-1 and g_agent.y == g_other.y:
            if e.BOMB_DROPPED:
                return DROPPED_BOMB_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x and g_agent.y == g_other.y-1:
            if e.BOMB_DROPPED:
                return DROPPED_BOMB_NEXT_TO_OTHER_AGENT
        if g_agent.x == g_other.x and g_agent.y == g_other.y+1:
            if e.BOMB_DROPPED:
                return DROPPED_BOMB_NEXT_TO_OTHER_AGENT


def unnecessary_bomb(state):
    if state is None or state['field'] is None:
        return
    if e.BOMB_EXPLODED and not e.KILLED_OPPONENT and not e.CRATE_DESTROYED:
        return UNNECESSARY_BOMB


            

def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        
        e.INVALID_ACTION: -1,
        e.CRATE_DESTROYED: 2,
        e.BOMB_DROPPED: 1,
        e.KILLED_SELF: -30,
        e.KILLED_OPPONENT: 10,
        e.SURVIVED_ROUND: 1,
        e.MOVED_DOWN: -.1,
        e.MOVED_LEFT: -.1,
        e.MOVED_RIGHT: -.1,
        e.MOVED_UP: -.1,
        e.WAITED: -.3,
        BOMB_FLED: 5,
        #IS_NEXT_TO_OTHER_AGENT: 2,
        WALKS_INTO_BOMB_RADIUS: -3,
        AGENT_MOVED_OUT_OF_BOMB_TILE: 3,
        MOVED_TOWARDS_CLOSEST_AGENT: 3,
        UNNECESSARY_BOMB: -11,#somehow in this setting too many invalid operations...
        #DROPPED_BOMB_NEXT_TO_OTHER_AGENT: 5
        
    }
    if is_subset(events, [e.BOMB_DROPPED, WALKS_INTO_BOMB_RADIUS]):
        events.remove(WALKS_INTO_BOMB_RADIUS)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum





