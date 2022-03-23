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

# todo make 2 frames available to agent

np.set_printoptions(linewidth=250)

register()

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='Training data output folder', required=True)
parser.add_argument('--n', help='Total number of data generation jobs', type=int, default=10)
parser.add_argument('--episodes', help='Total number of episodes to run per data generation job', type=int, default=10)

class Self:
    logger = logging.getLogger("Self")

@ray.remote
def work(out_folder, id, n_episodes=100):
    register()
    settings, agents = classic_with_opponents()
    env = gym.make("BomberGym-v6", args=settings, agents=agents)

    self = Self()
    setup(self)

    global_buffer = []
    for i in range(n_episodes):
        episode_buffer = []
        obs = env.reset()
        obs_previous = None
        orig_state = env.env.initial_state
        while True:
            action = act(self, orig_state)
            if action is None:
                action = "WAIT"
            action = ACTIONS.index(action)
            obs, rew, done, other = env.step(action)
            double_obs = None
            if obs is not None:
                if obs_previous is None:
                    double_obs = np.dstack((obs, obs))
                else:
                    double_obs = np.dstack((obs_previous, obs))
                new_data = (double_obs, action, rew)
                obs_previous = obs
                episode_buffer.append(new_data)
            orig_state = other["orig_state"]
            if done:
                if env.env.did_i_die() or not env.env.did_i_win():
                    print(f'Skipping nonoptimal episode {i} (Died: {env.env.did_i_die()}, Won: {env.env.did_i_win()})')
                    break
                print(f'Adding winning episode {i}')
                time = datetime.utcnow().isoformat(timespec='milliseconds')
                for frame_idx, frame in enumerate(episode_buffer):
                    filename = f'{out_folder}/{time}-ray-{id}-episode-{i}-frame-{frame_idx}.pickle.xz'
                    with lzma.open(filename, 'wb') as fd:
                        pickle.dump(frame, fd)
                break

if __name__ == '__main__':
    args = parser.parse_args()
    jobs = []
    os.makedirs(args.output, exist_ok=True)
    for i in range(args.n):
        jobs.append(work.remote(args.output, i, n_episodes=args.episodes))
        #jobs.append(work(args.output, i, n_episodes=args.episodes))

    ray.get(jobs)

