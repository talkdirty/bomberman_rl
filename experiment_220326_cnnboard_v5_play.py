import gym
from bombergym.scenarios import classic_with_opponents, classic_tournament
from bombergym.environments import register
import torch
import time
import numpy as np

from experiment_220322_resnet_model import CnnboardResNet

model = CnnboardResNet()
model.load_state_dict(torch.load("out_220326_supervised_resnet18_v5/model18.pth"))
model.eval()

register()
settings, agents = classic_tournament()
#settings, agents = classic_with_opponents()
#settings, agents = coin_heaven()

env = gym.make('BomberGym-v5', args=settings, agents=agents)

def detect_initial_configuration(obs):
    agent_frame = obs[4, :, :]
    x, y = (np.round(agent_frame) == 1).nonzero()
    x, y = x[0], y[0]
    config = None
    if x < 7 and y > 7:
        config = 'bottom-left'
    if x > 7 and y < 7:
        config = 'top-right'
    if x < 7 and y < 7:
        config = 'top-left'
    if x > 7 and y > 7:
        config = 'bottom-right'
    return config

def get_transposer(config):
    if config == 'top-left':
        return lambda model, input: model(torch.from_numpy(input).unsqueeze(0).to(torch.float32)).argmax().item()
    elif config == 'top-right':
        return transposer_lr
    elif config == 'bottom-left':
        return transposer_td
    else:
        return transposer_lrtd

def transposer_lr(model, inp):
    input_aug = inp[:, ::-1, :].copy()
    action = model(torch.from_numpy(input_aug).unsqueeze(0)).argmax().item()
    new_action_lr = None
    if action == 1: # Right
        new_action_lr = 3 # Left
    elif action == 3: # Left
        new_action_lr = 1 # Right
    else:
        new_action_lr = action
    return new_action_lr

def transposer_td(model, inp):
    input_aug = inp[:, :, ::-1].copy()
    action = model(torch.from_numpy(input_aug).unsqueeze(0)).argmax().item()
    new_action_ud = None
    if action == 0: # Up
        new_action_ud = 2 # down
    elif action == 2: # Down
        new_action_ud = 0 # Up
    else:
        new_action_ud = action
    return new_action_ud

def transposer_lrtd(model, inp):
    input_aug = inp[:, ::-1, ::-1].copy()
    action = model(torch.from_numpy(input_aug).unsqueeze(0)).argmax().item()
    new_action_udlr = None
    if action == 0: # Up
        new_action_udlr = 2 # down
    elif action == 2: # Down
        new_action_udlr = 0 # Up
    elif action == 1: # Right
        new_action_udlr = 3 # Left
    elif action == 3: # Left
        new_action_udlr = 1 # Right
    else:
        new_action_udlr = action
    return new_action_udlr

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
    
 
