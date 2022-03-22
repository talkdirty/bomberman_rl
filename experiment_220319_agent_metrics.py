# Compute Baseline metric of rule_based_agent
# Should be 25% if universe works
from datetime import datetime
import os
import argparse
import numpy as np
import pickle

import lzma
import ray
import gym
import logging
from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register
from bombergym.agent_code.rule_based_agent.callbacks import act, setup
from bombergym.settings import ACTIONS
import bombergym.original.events as e
import time

class Self:
    logger = logging.getLogger("Self")

@ray.remote
def work(n_episodes=100):
    register()
    settings, agents = classic_with_opponents()
    env = gym.make("BomberGym-v4", args=settings, agents=agents)

    self = Self()
    setup(self)
    
    won = 0
    lost = 0
    for i in range(n_episodes):
        episode_buffer = []
        obs = env.reset()
        # env.render()
        orig_state = env.env.initial_state
        while True:
            action = act(self, orig_state)
            if action is None:
                action = "WAIT"
            action = ACTIONS.index(action)
            obs, rew, done, other = env.step(action)
            # env.render()
            # time.sleep(.5)
            orig_state = other["orig_state"]
            if done:
                if env.env.did_i_win():
                    won += 1
                else:
                    lost += 1
                break
    return won, lost
    print(f'Won: {won}, Lost: {lost}, Winning fraction: {won/(won+lost)}')

if __name__ == '__main__':
    register()
    #for i in range(args.n):
        #jobs.append(work.remote(args.output, i, n_episodes=args.episodes))
    #jobs.append(work(args.output, 0, n_episodes=args.episodes))
    jobs = []
    for i in range(12):
        jobs.append(work.remote(n_episodes=100))
    data = ray.get(jobs)
    total_won, total_lost = 0, 0
    for won, lost in data:
        total_won += won
        total_lost += lost
    print(f'Total won: {total_won}, Total lost: {total_lost}, Total winning frac: {total_won/(total_won+total_lost)}')
