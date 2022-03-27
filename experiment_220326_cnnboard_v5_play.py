import gym
from bombergym.scenarios import classic_with_opponents, classic_tournament
from bombergym.environments import register
import torch
import time
import numpy as np

from experiment_220322_resnet_model import CnnboardResNet
from experiment_220326_cnnboard_v5_common import get_transposer, detect_initial_configuration

model = CnnboardResNet()
model.load_state_dict(torch.load("out_220326_supervised_resnet18_v5/model18.pth"))
model.eval()

register()
settings, agents = classic_tournament()
#settings, agents = classic_with_opponents()
#settings, agents = coin_heaven()

env = gym.make('BomberGym-v5', args=settings, agents=agents)

obs = env.reset()
initial_config = detect_initial_configuration(obs)
transposer = get_transposer(initial_config)
print(f'Config: {initial_config}')
time.sleep(1)
env.render()
while True:
    action = transposer(model, obs.astype(np.float32))
    obs, rew, done, other = env.step(action)
    if not done:
        feature_info = other["features"] if "features" in other else None
        env.render(events=other["events"], rewards=rew, other=feature_info)
        time.sleep(.5)
    else:
        print(other["events"], f"Reward: {rew}")
        break
    
 
