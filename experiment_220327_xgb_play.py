import gym
from bombergym.scenarios import classic_with_opponents, classic_tournament, coin_heaven
from bombergym.environments import register
import time
import numpy as np
from bombergym.environments.plain.features import state_to_gym
from bombergym.environments.plain.navigation import Tile
import bombergym.settings as s

import xgboost as xgb
import torch

model = xgb.Booster({'nthread': 4})
model.load_model('bst0001.model')

def get_available_actions(orig_state, gym_state):
    available_actions = ['WAIT']
    if orig_state['self'][2]:
        available_actions.append('BOMB')
    agent_x, agent_y = orig_state['self'][3]
    if Tile.walkable(gym_state[agent_y, agent_x - 1]):
        available_actions.append('LEFT')
    if Tile.walkable(gym_state[agent_y, agent_x + 1]):
        available_actions.append('RIGHT')
    if Tile.walkable(gym_state[agent_y + 1, agent_x]):
        available_actions.append('DOWN')
    if Tile.walkable(gym_state[agent_y - 1, agent_x]):
        available_actions.append('TOP')
    return available_actions

register()
#settings, agents = classic_tournament()
#settings, agents = classic_with_opponents()
settings, agents = coin_heaven()

env = gym.make('BomberGym-v3', args=settings, agents=agents)

obs = env.reset()
orig_obs = env.env.initial_state
time.sleep(1)
env.render()
while True:
    actions_avail = get_available_actions(orig_obs, state_to_gym(orig_obs))
    actions_avail = [s.ACTIONS.index(a) for a in actions_avail]
    action_probs = model.predict(xgb.DMatrix(obs[None, ...]))
    action_probs_sorted = action_probs.argsort().squeeze()
    action = s.ACTIONS.index('WAIT')
    for i in range(len(action_probs_sorted)):
        action_candidate = action_probs_sorted[i]
        if action_candidate in actions_avail:
            action = action_candidate
            break
    obs, rew, done, other = env.step(action)
    if not done:
        feature_info = other["features"] if "features" in other else None
        env.render(events=other["events"], rewards=rew, other=feature_info)
        time.sleep(.5)
    else:
        print(other["events"], f"Reward: {rew}")
        break
    
 
