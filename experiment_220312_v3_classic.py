import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from bombergym.scenarios import classic

from bombergym.environments import register
from experiments.monitoring import CustomCallback

register()

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', required=True) 
parser.add_argument('--experiment', help='Description of the experiment', required=True) 
parser.add_argument('--total-timesteps', required=False, type=int, default=1000000) 

args = parser.parse_args()

settings, agents = classic()

env = make_vec_env("BomberGym-v3", n_envs=4, env_kwargs={'args': settings, 'agents': agents})
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"{args.checkpoint_dir}/{args.experiment}/")
callback = CustomCallback(args.checkpoint_dir, args.experiment)
model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=args.experiment)
model.save("bombermodel")