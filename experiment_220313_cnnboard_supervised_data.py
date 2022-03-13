import os
import argparse
import pickle

import ray
import gym
import logging
from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register
from bombergym.agent_code.rule_based_agent.callbacks import act, setup
from bombergym.settings import ACTIONS
import bombergym.original.events as e

register()

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='Training data output pickle', required=True)
parser.add_argument('--n', help='Total number of data generation jobs', type=int, default=10)
parser.add_argument('--episodes', help='Total number of episodes to run per data generation job', type=int, default=10)

class Self:
    logger = logging.getLogger("Self")

@ray.remote
def work(output_name, n_episodes=100):
    register()
    settings, agents = classic_with_opponents()
    env = gym.make("BomberGym-v4", args=settings, agents=agents)

    self = Self()
    setup(self)

    global_buffer = []
    for i in range(n_episodes):
        episode_buffer = []
        obs = env.reset()
        orig_state = env.env.initial_state
        while True:
            action = act(self, orig_state)
            if action is None:
                action = "WAIT"
            action = ACTIONS.index(action)
            old_obs = obs
            obs, rew, done, other = env.step(action)
            episode_buffer.append((old_obs, action, rew, obs))
            orig_state = other["orig_state"]
            if done:
                if env.env.did_i_die():
                    print(f'Skipping nonoptimal episode {i}')
                    break
                print(f'Adding winning episode {i}')
                global_buffer.append(episode_buffer)
                break
    with open(output_name, 'wb') as fd:
        pickle.dump(global_buffer, fd)
    return global_buffer
    

if __name__ == '__main__':
    args = parser.parse_args()
    jobs = []
    os.makedirs(args.output, exist_ok=True)
    for i in range(args.n):
        jobs.append(work.remote(f"{args.output}/data{i}.pckl", n_episodes=args.episodes))
    ray.get(jobs)

