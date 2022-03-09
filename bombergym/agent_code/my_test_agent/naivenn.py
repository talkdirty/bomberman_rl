import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from settings import ACTIONS

TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
MEMORY_SIZE = 1000
BATCH_SIZE = 20
EXPLORATION_RATE = .1
GAMMA = .95

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.9995
LEARNING_RATE = 1e-3

USE_CUDA = torch.cuda.is_available()


class NaiveNN(nn.Module):
    def __init__(self, input_size, output_size=len(ACTIONS)):
        super(NaiveNN, self).__init__()
        self.inp = nn.Linear(input_size, 256)
        self.l1 = nn.Linear(256, 256)
        self.out = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.inp(x)
        x = F.relu(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.out(x)
        return x

def act(self, state):
    if np.random.rand() < self.exploration_rate:
        return ACTIONS[random.randrange(len(ACTIONS))]
    self.model.eval()
    if USE_CUDA:
        state = state.cuda()
    q_values = self.model(state)
    return ACTIONS[q_values.argmax()]

#def remember(self, state, action, reward, next_state, done):
#    self.memory.append((
#        state_to_features(state), 
#        action, 
#        reward, 
#        state_to_features(next_state), 
#        done
#        ))

def fit_model(self, X, y):
    self.model.train() # Put model in training mode
    self.optimizer.zero_grad()
    if USE_CUDA:
        X, y = X.cuda(), y.cuda()
    with torch.set_grad_enabled(True):
        outputs = self.model(X)
        loss = self.loss(outputs, y)
        loss.backward()
        self.optimizer.step()

def experience_replay(self):
    if len(self.memory) < BATCH_SIZE:
        return
    batch = random.sample(self.memory, BATCH_SIZE)
    for state, action, reward, state_next, terminal in batch:
        if state is None:
            continue
        if USE_CUDA:
            state_next, state = state_next.cuda(), state.cuda()
        action_id = ACTIONS.index(action)
        q_update = reward
        if not terminal:
            model_pass = self.model(state_next)
            q_update = (reward + GAMMA * model_pass.max())
        q_values = self.model(state)
        q_values[action_id] = q_update
        fit_model(self, state, q_values)
    self.exploration_rate *= EXPLORATION_DECAY
    self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)