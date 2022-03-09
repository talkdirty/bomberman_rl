import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

import bombergym.settings as s
from bombergym.environments import register
from bombergym.environments.callbacks import CustomCallback

register()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', required=True) 
parser.add_argument('--total-timesteps', required=False, type=int, default=1000000) 

args = parser.parse_args()

# Instantiate Custom Callback for checkpointing and evaling model
if args.checkpoint_dir:
    callback = CustomCallback(args.checkpoint_dir)
else:
    callback = None

# Way to avoid tedious argparse
class BombeRLeSettings:
    command_name = 'play'
    my_agent = None
    agents = [
        'gym_surrogate_agent', # Important, needs to be first. Represents our agent
        #'random_agent', # Possibility to add other agents here
    ]
    train = 1
    continue_without_training = False
    #scenario = 'coin-heaven'
    scenario = 'classic'
    seed = None
    n_rounds = 10 # Has no effect
    save_replay = False # Has no effect
    match_name = None # ?
    silence_errors = False
    skip_frames = False # No effect
    no_gui = False # No effect
    turn_based = False # No effect
    update_interval = .1 # No effect
    log_dir = './logs'
    save_stats = False # No effect ?
    make_video = False # No effect

bomber = BombeRLeSettings()

# Setup agents
agents = []
if bomber.train == 0 and not bomber.continue_without_training:
    bomber.continue_without_training = True
if bomber.my_agent:
    agents.append((bomber.my_agent, len(agents) < bomber.train))
    bomber.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
for agent_name in bomber.agents:
    agents.append((agent_name, len(agents) < bomber.train))

env = make_vec_env("BomberGym-v0", n_envs=4, env_kwargs={'args': bomber, 'agents': agents})
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=args.total_timesteps, callback=callback)
model.save("bombermodel")