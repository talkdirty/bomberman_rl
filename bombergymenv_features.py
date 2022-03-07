import numpy as np

from bombergymenv import BombeRLeWorld
from events import BOMB_DROPPED, BOMB_EXPLODED, MOVED_DOWN, MOVED_LEFT, MOVED_RIGHT, MOVED_UP
import settings as s
import events as e

import gym
from gym import spaces

from agent_code.gym_surrogate_agent.features import state_to_gym

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
    if num_crates > 0 and is_subset(events, [BOMB_EXPLODED]) and not is_suicide(events):
        return num_crates

BOMB_FLED = "BOMB_FLED"
WALKS_INTO_BOMB_RADIUS = "WALKS_INTO_BOMB_RADIUS"
AGENT_MOVED_OUT_OF_BOMB_TILE = "AGENT_MOVED_OUT_OF_BOMB_TILE"

def bomb_pathfinding_grid(state):
    grid = np.zeros_like(state['field'])
    grid[state['field'] == 1] = -1 # Crates are unwalkable
    grid[state['field'] == -1] = -1 # Walls are unwalkable
    grid[state['field'] == 0] = 1 # Floor is walkable
    g = Grid(matrix=grid)
    return g

# TODO refactor
def agent_on_bomb(state):
    if state['field'] is None:
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
    if state['field'] is None:
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

class BombeRLeWorldFeatureEng(BombeRLeWorld, gym.Env):
    """Adapted environment with MORE feature engineering"""

    def __init__(self, args, agents):
      super().__init__(args, agents)
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Dict({
          'field': spaces.MultiDiscrete([5] * (s.ROWS * s.COLS)),
          'bombs': spaces.MultiDiscrete([s.BOMB_TIMER + 1] * (s.ROWS * s.COLS)),
          'explosions': spaces.MultiDiscrete([s.EXPLOSION_TIMER + 1] * (s.ROWS * s.COLS)),
          'coins': spaces.MultiDiscrete([2] * (s.ROWS * s.COLS)),
          'other_bombs': spaces.MultiDiscrete([2] * 3),
      })
    
    def reset(self):
        """Gym API reset"""
        self.new_round()
        orig_state = self.get_state_for_agent(self.agents[0])
        return state_to_gym(orig_state)

    def compute_extra_events(self, old_state: dict, new_state: dict, action):
        events = []
        bomb_sight = bomb_fled_event(old_state, new_state, action)
        if bomb_sight is not None:
            events.append(bomb_sight)
        agent_bomb = agent_moved_out_of_bomb_tile(old_state, new_state, action)
        if agent_bomb is not None:
            events.append(agent_bomb)
        return events

    def step(self, action):
        action_orig = s.ACTIONS[action]
        # Treat our own agent specially and dispatch its action
        # WARNING alters game logic a bit, as we are usually not
        # executed first (TODO relevant?)
        self.perform_agent_action(self.agents[0], action_orig)
        # Hook into original logic and dispatch world update
        events = self.do_step()
        orig_state = self.get_state_for_agent(self.agents[0])
        # Provide facility for computing extra events with altered control
        # flow, similar to train:game_events_occured in callback version
        more_events = self.compute_extra_events(self.last_state, orig_state, action)
        self.last_state = orig_state
        if more_events:
            events = events + more_events
        own_reward = reward_from_events(events)
        done = self.time_to_stop()
        other = {"events": events}
        return state_to_gym(orig_state), own_reward, done, other

    def close(self):
        pass