import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import time

from .cnnboard_features import state_to_gym
from .experiment_220322_cnnboard_val_metric import detect_initial_configuration, get_transposer
from .experiment_220322_resnet_model import CnnboardResNet


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    device = torch.device("cpu")

    model_start = time.time()
    model = CnnboardResNet().to(device)
    model.load_state_dict(torch.load("model50.pth", map_location=device))
    model.eval()
    model_end = time.time()

    self.logger.info(f"Loaded model in {model_end-model_start}s.")

    self.initial_config = None
    self.model = model
    self.logger.info("Loaded model")

    history_buffer = deque(maxlen=3)
    self.current_round = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        self.logger.info("Resetting self")
        self.history_buffer = deque(maxlen=3)
        self.current_round = game_state["round"]
    if len(self.history_buffer) != 3:
        self.history_buffer.append(game_state)
        self.history_buffer.append(game_state)
        self.history_buffer.append(game_state)
    else:
        self.history_buffer.append(game_state)
    obs = state_to_gym(game_state, self.history_buffer)
    if not self.initial_config:
        self.initial_config = detect_initial_configuration(obs)
        self.logger.info(f"detected initial configuration: {self.initial_config}")
    transposer = get_transposer(self.initial_config)

    action_start = time.time()
    action = transposer(self.model, obs.astype(np.float32))
    action_end = time.time()
    self.logger.info(f'Taking action {action} took {action_end-action_start}s.')
    return ACTIONS[action]
