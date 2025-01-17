from os import stat
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

    def __init__(self, args, agents, state_fn=state_to_gym, reward_fn=reward_from_events):
      super().__init__(args, agents)
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Box(low=-1, high=1, shape=(25,))
      self.state_fn = state_fn
      self.reward_fn = reward_fn

    def reset(self):
        """Gym API reset"""
        self.new_round()
        orig_state = self.get_state_for_agent(self.agents[0])
        self.initial_state = orig_state
        return state_to_gym(orig_state)

    def compute_extra_events(self, old_state: dict, new_state: dict, action):
        return []

    def step(self, action):
        action_orig = s.ACTIONS[action]
        # Treat our own agent specially and dispatch its action
        # WARNING alters game logic a bit, as we are usually not
        # executed first (TODO relevant?)
        # self.perform_agent_action(self.agents[0], action_orig)
        # Hook into original logic and dispatch world update
        events = self.do_step(action_orig)
        orig_state = self.get_state_for_agent(self.agents[0])
        # Provide facility for computing extra events with altered control
        # flow, similar to train:game_events_occured in callback version
        more_events = self.compute_extra_events(self.last_state, orig_state, action)
        self.last_state = orig_state
        if more_events:
            events = events + more_events
        own_reward = self.reward_fn(events)
        done = self.time_to_stop()

        feats = self.state_fn(orig_state)
        other = {"events": events, "features": feats, "orig_state": orig_state}
        return feats, own_reward, done, other

    def did_i_win(self):
        enemy_scores = []
        my_score = 0
        for agent in self.agents:
            if agent.code_name == "gym_surrogate_agent":
                my_score = agent.score
            else:
                enemy_scores.append(agent.score)
        # print(f'my score: {my_score}, enemy scores: {enemy_scores}')
        return True
        if len(enemy_scores) > 0 and my_score > max(enemy_scores):
            return True
        else:
            return False

    def did_i_die(self):
        died = True
        for agent in self.active_agents:
            if agent.code_name == "gym_surrogate_agent":
                died = False
                break
        return died

    def close(self):
        pass
