import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from bombergym.scenarios import coin_heaven

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

settings, agents = coin_heaven()

env = make_vec_env("BomberGym-v1", n_envs=4, env_kwargs={'args': settings, 'agents': agents})
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=args.total_timesteps, callback=callback)
model.save("bombermodel")