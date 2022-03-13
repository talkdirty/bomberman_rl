import gym
import getch
from stable_baselines3.common.env_checker import check_env
from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register
import torch
import time
import numpy as np

from experiment_220313_cnnboard_train import CnnBoardNetwork

register()
settings, agents = classic_with_opponents()

env = gym.make('BomberGym-v4', args=settings, agents=agents)

model = CnnBoardNetwork()
model.load_state_dict(torch.load("model.pth"))

obs = env.reset()
env.render()
while True:
    inp = torch.from_numpy(obs.swapaxes(0,2).astype(np.float32)).unsqueeze(0)
    print(inp.shape)
    action = model(inp)
    obs, rew, done, other = env.step(action.argmax().item())
    if not done:
        feature_info = other["features"] if "features" in other else None
        env.render(events=other["events"], rewards=rew, other=feature_info)
        time.sleep(.5)
    else:
        print(other["events"], f"Reward: {rew}")
        break
    
 
