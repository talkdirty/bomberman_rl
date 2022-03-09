import argparse

import pickle
import bombergym.render
import time

parser = argparse.ArgumentParser()
parser.add_argument('--recording', required=True) 
parser.add_argument('--speed', required=False, default=.5) 
parser.add_argument('--speed-end', required=False, default=2) 

args = parser.parse_args()

with open(args.recording, 'rb') as fd:
    recording = pickle.load(fd)

for episode in recording:
    for step in episode:
        obs, action, rew, done, other = step
        orig_state = other["orig_state"]
        if orig_state is not None:
            bombergym.render.render(orig_state, events=other["events"], rewards=rew)
        time.sleep(args.speed)
    time.sleep(args.speed_end)