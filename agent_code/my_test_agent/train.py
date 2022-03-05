import random
import numpy as np
from collections import namedtuple, deque

import pickle
from typing import List
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler

import events as e
from . import naivenn
from .naivenn import NaiveNN
from .naivenn import MEMORY_SIZE, EXPLORATION_MAX, LEARNING_RATE, USE_CUDA

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    torch.set_grad_enabled(False)
    self.model = NaiveNN(313)
    if USE_CUDA:
        self.model = self.model.cuda()
    self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
    self.loss = torch.nn.MSELoss()
    self.memory = deque(maxlen=MEMORY_SIZE)
    self.exploration_rate = EXPLORATION_MAX

    self.total_steps = 0
    self.total_rewards = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    reward = reward_from_events(self, events)
    naivenn.remember(self, old_game_state, self_action, reward, new_game_state, False)

    self.total_steps += 1
    self.total_rewards += reward

    self.last_state = new_game_state
    naivenn.experience_replay(self)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    reward = reward_from_events(self, events)
    naivenn.remember(self, self.last_state, last_action, reward, last_game_state, True)

    self.total_steps += 1
    self.total_rewards += reward

    print('Total Steps, Rewards', self.total_steps, self.total_rewards, self.total_rewards / self.total_steps)

    self.total_steps, self.total_rewards = 0, 0

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 8,
        e.COIN_FOUND: 5,
        e.KILLED_SELF: -5,
        e.WAITED: -1,
        e.MOVED_LEFT: -.05,
        e.MOVED_RIGHT: -.05,
        e.MOVED_UP: -.05,
        e.MOVED_DOWN: -.05,
        e.INVALID_ACTION: -3,
        e.KILLED_OPPONENT: 10,
        e.OPPONENT_ELIMINATED: 15,
        e.SURVIVED_ROUND: 10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
