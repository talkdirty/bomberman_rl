import torch
import ray
from bombergym.scenarios import classic, classic_with_opponents, coin_heaven
from bombergym.environments import register
import gym
import numpy as np

from experiment_220324_deepmind_arch import DeepmindAtariCNN, DeepmindAtariCNNDeep

def detect_initial_configuration(obs):
    agent_frame = obs[4, :, :]
    x, y = (agent_frame == 1).nonzero()
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
        return lambda model, input: model(torch.from_numpy(input).unsqueeze(0)).argmax().item()
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

def validate_model(model):
    # Load model
    register()
    settings, agents = classic_with_opponents()
    env = gym.make("BomberGym-v4", args=settings, agents=agents)

    won, lost = 0, 0
    obs = env.reset()
    initial_config = detect_initial_configuration(obs)
    transposer = get_transposer(initial_config)
    prev_obs = None
    while True:
        if prev_obs is None:
            double_obs = np.dstack((obs, obs))
        else:
            double_obs = np.dstack((prev_obs, obs))
        action = transposer(model, double_obs.astype(np.float32))
        prev_obs = obs
        obs, rew, done, other = env.step(action)
        if done:
            if env.env.did_i_win():
                won += 1
            else:
                lost += 1
            break
    return won, lost

def validate(model_state_dict, n_episodes):
    device = torch.device("cpu")
    model = DeepmindAtariCNNDeep().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    stats = []
    for i in range(n_episodes):
        stats.append(validate_model(model)) 
    total_won, total_lost = 0, 0
    for won, lost in stats:
        total_won += won
        total_lost += lost
    return total_won / (total_won + total_lost)