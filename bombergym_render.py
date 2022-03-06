import numpy as np
import settings

bomb = '💣'
money = '💵'
wall = '🧱'
box = '📦'
robot = '🤖'
self = '💩'
collision = '💥'
space = '➖'

def render(state, events=None, rewards=None, clear=True):
    if clear:
        print("\033c", end="")
    field = np.zeros((settings.ROWS, settings.COLS), dtype=str)
    field[:, :] = space

    field[(state['field'] == 1).T] = box
    field[(state['field'] == -1).T] = wall
    for coin in state['coins']:
        field[coin[1], coin[0]] = money
    field[state['self'][3][1], state['self'][3][0]] = self
    for other in state['others']:
        field[other[3][1], other[3][0]] = robot
    for b in state['bombs']:
        field[b[0][1], b[0][0]] = bomb
    field[(state['explosion_map'] != 0).T] = collision

    for row in field:
        for col in row:
            print(col, end='')
        print()
    print(f'Step: {state["step"]}/{state["round"]}')
    if events:
        print(f'Events: {", ".join(events)}')
    if rewards:
        print(f'Rewards: {rewards}')