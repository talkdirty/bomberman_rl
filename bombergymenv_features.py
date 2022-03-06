import numpy as np

from bombergymenv import BombeRLeWorld
from events import BOMB_EXPLODED
import settings as s
import events as e

import gym
from gym import spaces

from agent_code.gym_surrogate_agent.features import state_to_gym

def is_subset(set, subset):
    return all(x in set for x in subset)

def is_suicide(events):
    return is_subset(events, [e.BOMB_EXPLODED, e.KILLED_SELF, e.GOT_KILLED])

def is_good_bomb_placement(events):
    num_crates = events.count(e.CRATE_DESTROYED)
    if num_crates > 0 and is_subset(events, [BOMB_EXPLODED]) and not is_suicide(events):
        return num_crates

def reward_from_events(events) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    if is_suicide(events):
        return -2
    crates_destroyed = is_good_bomb_placement(events)
    if crates_destroyed is not None:
        return 2
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.INVALID_ACTION: -1,
        e.KILLED_OPPONENT: 5,
        e.SURVIVED_ROUND: 1,
    }
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

    def step(self, action):
        action_orig = s.ACTIONS[action]
        self.perform_agent_action(self.agents[0], action_orig)
        events = self.do_step()
        own_reward = reward_from_events(events)
        orig_state = self.get_state_for_agent(self.agents[0])
        done = self.time_to_stop()
        other = {"events": events}
        return state_to_gym(orig_state), own_reward, done, other

    def close(self):
        pass