import numpy as np

from bombergymenv import BombeRLeWorld
import settings as s
import events as e

import gym
from gym import spaces

from agent_code.gym_surrogate_agent.features import agent_moved_out_of_bomb_tile, bomb_fled_event, reward_from_events, state_to_gym


class BombeRLeWorldFeatureEng(BombeRLeWorld, gym.Env):
    """Adapted environment with MORE feature engineering"""

    def __init__(self, args, agents):
      super().__init__(args, agents)
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Dict({
          #'field': spaces.MultiDiscrete([6] * (s.ROWS * s.COLS)),
          'field': spaces.Box(low=0, high=8, shape=(s.ROWS, s.COLS)),
          #'bombs': spaces.MultiDiscrete([s.BOMB_TIMER + 1] * (s.ROWS * s.COLS)),
          # TODO replace with "danger" situational awareness
          #'explosions': spaces.MultiDiscrete([s.EXPLOSION_TIMER + 1] * (s.ROWS * s.COLS)),
          #'coins': spaces.MultiDiscrete([2] * (s.ROWS * s.COLS)),
          'bomb_awareness': spaces.Box(low=np.array([-s.BOMB_POWER, -s.BOMB_POWER]), high=np.array([s.BOMB_POWER, s.BOMB_POWER])),
          'bomb_on': spaces.Discrete(2)
          #'other_bombs': spaces.MultiDiscrete([2] * 3),
          #'others': spaces.MultiDiscrete([2] * (s.ROWS * s.COLS))
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

        #TODO temp some features here:
        log_features = {
        }
        feats = state_to_gym(orig_state)
        other = {"events": events, "features": feats}
        return feats, own_reward, done, other

    def close(self):
        pass