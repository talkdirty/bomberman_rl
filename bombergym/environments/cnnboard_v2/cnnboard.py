from collections import deque

from bombergym.environments.base import BombeRLeWorld
import bombergym.settings as s

import gym
from gym import spaces

from .features import state_to_gym
from bombergym.environments.manhattan_v2.rewards import reward_from_events
import numpy
numpy.set_printoptions(linewidth=250)

class BomberGymCnnBoard(BombeRLeWorld, gym.Env):

    def __init__(self, args, agents, state_fn=state_to_gym, reward_fn=reward_from_events):
      super().__init__(args, agents)
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Box(low=-1, high=1, shape=(5, s.COLS, s.ROWS))
      self.state_fn = state_fn
      self.reward_fn = reward_fn

      self.previous_states = deque(maxlen=3)

    def reset(self):
        """Gym API reset"""
        self.new_round()
        orig_state = self.get_state_for_agent(self.agents[0])
        self.initial_state = orig_state
        self.previous_states.append(orig_state)
        self.previous_states.append(orig_state)
        self.previous_states.append(orig_state)
        return state_to_gym(orig_state, self.previous_states)

    def compute_extra_events(self, old_state: dict, new_state: dict, action):
        return []

    def did_i_win(self):
        enemy_scores = []
        my_score = 0
        for agent in self.agents:
            if agent.code_name == "gym_surrogate_agent":
                my_score = agent.score
            else:
                enemy_scores.append(agent.score)
        # print(f'my score: {my_score}, enemy scores: {enemy_scores}')
        if my_score > max(enemy_scores):
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

    def step(self, action):
        action_orig = s.ACTIONS[action]
        # Hook into original logic and dispatch world update
        events = self.do_step(action_orig)
        orig_state = self.get_state_for_agent(self.agents[0])
        # Provide facility for computing extra events with altered control
        # flow, similar to train:game_events_occured in callback version
        more_events = self.compute_extra_events(self.last_state, orig_state, action)
        self.last_state = orig_state
        self.previous_states.append(orig_state)
        if more_events:
            events = events + more_events
        own_reward = self.reward_fn(events)
        done = self.time_to_stop()

        feats = self.state_fn(orig_state, self.previous_states)
        other = {"events": events, "features": feats, "orig_state": orig_state}
        return feats, own_reward, done, other

    def close(self):
        pass
