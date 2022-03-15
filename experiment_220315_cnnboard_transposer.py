import gym
import getch
from stable_baselines3.common.env_checker import check_env
from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register
import torch
import time
import numpy as np

np.set_printoptions(linewidth=120)

from experiment_220313_cnnboard_train import CnnBoardNetwork

register()
settings, agents = classic_with_opponents()

env = gym.make('BomberGym-v4', args=settings, agents=agents)

obs = env.reset()

def detect_initial_configuration(obs):
    agent_frame = obs[4, :, :]
    x, y = (agent_frame == 1).nonzero()
    x, y = x[0], y[0]
    config = None
    if x < 7 and y > 7:
        config = 'top-right'
    if x > 7 and y < 7:
        config = 'bottom-left'
    if x < 7 and y < 7:
        config = 'top-left'
    if x > 7 and y > 7:
        config = 'bottom-right'
    return config

def get_transposer(config):
    if config == 'top-left':
        return lambda model, input: model(input)
    elif config == 'top-right':
        # Flip Lr
        return transposer_lr
    elif config == 'bottom-left':
        return transposer_td
    else:
        return transposer_lrtd

def transposer_lr(model, input):
    input_aug = input[:, :, ::-1]
    action = model(input_aug).argmax().item()
    new_action_lr = None
    if action == 1: # Right
        new_action_lr = 3 # Left
    elif action == 3: # Left
        new_action_lr = 1 # Right
    else:
        new_action_lr = action
    return new_action_lr

def transposer_td(model, input):
    input_aug = old_obs[:, ::-1, :]
    action = model(input_aug).argmax().item()
    new_action_ud = None
    if action == 0: # Up
        new_action_ud = 2 # down
    elif action == 2: # Down
        new_action_ud = 0 # Up
    else:
        new_action_ud = action
    return new_action_ud

def transposer_lrtd(model, input):
    input_aug = old_obs[:, ::-1, ::-1]
    action = model(input_aug).argmax().item()
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
    return action

def make_augmentations(old_obs, action, rew, obs):
    flipped_obs_lr = old_obs[:, :, ::-1]
    return [
            (old_obs, action, rew, obs), 
            (flipped_obs_lr, new_action_lr, rew, obs[:, :, ::-1] if obs is not None else None),
            (flipped_obs_ud, new_action_ud, rew, obs[:, ::-1, :] if obs is not None else None),
            (flipped_obs_udlr, new_action_udlr, rew, obs[:, ::-1, ::-1] if obs is not None else None),
            ]


transposer = get_transposer(initial_config)

env.render()
while True:
    inp = torch.from_numpy(obs.swapaxes(0,2).astype(np.float32)).unsqueeze(0)
    action = transposer(model, inp)
    obs, rew, done, other = env.step(action)
    if not done:
        feature_info = other["features"] if "features" in other else None
        env.render(events=other["events"], rewards=rew, other=feature_info)
        time.sleep(.5)
    else:
        print(other["events"], f"Reward: {rew}")
        break
    
 
