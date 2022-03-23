from bombergym.environments.base import BombeRLeWorld
import bombergym.settings as s
import numpy as np
import gym
from gym import spaces

from .features import state_to_gym
from .rewards_murder import reward_from_events, unnecessary_bomb
from .rewards_murder import moved_towards_agent, dropped_bomb_next_to_other

class BomberGymReduced(BombeRLeWorld, gym.Env):
    """
    Much more simplified, low dimensional version of the environment
    * Observation space is a low dimensional vector
      1. Danger (r, l, t, b), high if close to bomb or explosion
      2. Coin (r, l, t, b), high if coin is nearby
      3. Crate (r, l, t, b), high if close to a crate
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
        if new_state is not None:
            crates = np.sum(new_state['field'] == 1)
            if moved_towards_agent(old_state, new_state) and crates < 130:
                return [moved_towards_agent(old_state, new_state), unnecessary_bomb(new_state)]#moved_towards_agent(old_state, new_state), dropped_bomb_next_to_other(new_state)
            elif unnecessary_bomb(new_state):
                return [unnecessary_bomb(new_state)]
            else:
                return None
        else:
            return None


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