import gym

import settings as s
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

# Way to avoid tedious argparse
class BombeRLeSettings:
    command_name = 'play'
    my_agent = None
    agents = [
        'gym_surrogate_agent', # Important, needs to be first. Represents our agent
        'random_agent', # Possibility to add other agents here
    ]
    train = 1
    continue_without_training = False
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

args = BombeRLeSettings()

# Setup agents
agents = []
if args.train == 0 and not args.continue_without_training:
    args.continue_without_training = True
if args.my_agent:
    agents.append((args.my_agent, len(agents) < args.train))
    args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
for agent_name in args.agents:
    agents.append((agent_name, len(agents) < args.train))

gym.envs.register(
    id='BomberGym-v0',
    entry_point='bombergymenv:BombeRLeWorld',
    max_episode_steps=401,
    kwargs={ 'args': args, 'agents': agents }
)

env = make_vec_env("BomberGym-v0", n_envs=4)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("bombermodel")