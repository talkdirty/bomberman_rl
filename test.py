import unittest

import numpy as np
import gym
from stable_baselines3.common.env_checker import check_env
from bombergym.environments.manhattan.features import pathfinding_distance

from bombergym.scenarios import coin_heaven
from bombergym.environments import register
from bombergym.environments.plain.navigation import Tile as T

register()


class TestBomberGymV2(unittest.TestCase):
    def setUp(self):
        settings, agents = coin_heaven()
        self.env = gym.make('BomberGym-v2', args=settings, agents=agents)

    def test_check_env_works(self):
        check_env(self.env)

    def test_pathfinding_coin(self):
        grid = np.array([
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
            [T.WALL, T.EMPTY, T.EMPTY, T.WALL, T.WALL],
            [T.WALL, T.EMPTY, T.EMPTY, T.COIN, T.WALL],
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
        ])
        myself = (1, 1)
        dist = pathfinding_distance(grid, myself, T.COIN)
        self.assertEqual(dist, 4, "Path (including myself and coin tile) should be four moves long")
        myself_right_next_to_coin = (2, 2)
        dist = pathfinding_distance(grid, myself_right_next_to_coin, T.COIN)
        self.assertEqual(dist, 2)

    def test_pathfinding_unreachable(self):
        grid = np.array([
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
            [T.WALL, T.EMPTY, T.WALL, T.WALL, T.WALL],
            [T.WALL, T.EMPTY, T.WALL, T.COIN, T.WALL],
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
            [T.WALL, T.WALL, T.WALL, T.WALL, T.WALL],
        ])
        myself = (1, 1)
        dist = pathfinding_distance(grid, myself, T.COIN)
        self.assertEqual(dist, np.inf)