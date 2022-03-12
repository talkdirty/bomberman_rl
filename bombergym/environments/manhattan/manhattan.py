from bombergym.environments.base import BombeRLeWorld
import bombergym.settings as s

import gym
from gym import spaces

from bombergym.environments.reduced.rewards import reward_from_events
from .features import state_to_gym

class BomberGymReducedManhattanNorm(BombeRLeWorld, gym.Env):
    """
    Like reduced environment, but instead of "stupid" line-of-sight level
    features, indicate the manhattan distance to the closest <thing> in
    feature space.
    * Action space are the usual set of discrete actions.
    * Some simple rewards are given.
    """

    def __init__(self, args, agents):
      super().__init__(args, agents)
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Box(low=-1, high=1, shape=(25,))

    def reset(self):
        """Gym API reset"""
        self.new_round()
        orig_state = self.get_state_for_agent(self.agents[0])
        return state_to_gym(orig_state)

    def compute_extra_events(self, old_state: dict, new_state: dict, action):
        return []

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

        feats = state_to_gym(orig_state)
        other = {"events": events, "features": feats, "orig_state": orig_state}
        return feats, own_reward, done, other

    def close(self):
        pass