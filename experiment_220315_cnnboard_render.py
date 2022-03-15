# Render CNNBoard in order to verify the data augmentations are correct.
import os
import lzma
import pickle
import sys
import getch
import numpy as np

import bombergym.settings as s

np.set_printoptions(threshold=np.inf)

bomb = '💣'
money = '💵'
wall = '🧱'
box = '📦'
robot = '🤖'
self = '💩'
collision = '💥'
space = '➖'

def render(data):
    old_obs, action, rew, new_obs = data

    field = np.zeros((17,17), dtype=str)
    field[:, :] = space
    
    field[(old_obs[0, :, :] == 1).T] = box
    field[(old_obs[1, :, :] != 0).T] = collision
    field[(old_obs[2, :, :] != 0).T] = wall
    field[(old_obs[3, :, :] != 0).T] = money
    field[(old_obs[4, :, :] == 1).T] = self
    field[(old_obs[4, :, :] == -1).T] = robot
    for row in field:
        for col in row:
            print(col, end='')
        print()
    print(f'Action: {s.ACTIONS[action]}')

if __name__ == '__main__':
    DIR = 'test333'
    d = os.listdir(DIR)
    ray = '1'
    episode = sys.argv[1]
    frame = sys.argv[2]
    substr = f'ray-{ray}-episode-{episode}-frame-{frame}.pickle'
    matching = [s for s in d if substr in s]
    print(f'n matching: {len(matching)}')
    for f in matching:
        print(f)
        with lzma.open(f'{DIR}/{f}', 'rb') as fd:
            data = pickle.load(fd)
        render(data)
        break
