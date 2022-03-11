# bomberman_rl - BomberGym

Bombergym features a rewrite of the original bomberman_rl game engine in order to be compatible with [OpenAI Gym](https://gym.openai.com/). By providing a gym compatible environment of bomberman, we hope to make our lives a bit easier: 

1. We want to be able to quickly experiment and iterate on cutting edge RL algorithms - many Deep RL libraries, for example [stable_baselines3](https://stable-baselines3.readthedocs.io/en/master/), can simply be directly plugged into an existing gym environment.
2. We can use the existing ecosystem and libraries to avoid having to write boilerplate. For example, we can use `make_vec_env` from `stable_baselines3` to create vectorized environments and train our agent in parallel without having to worry how to do this with the existing game engine.
3. We can make development and collaboration easier and more streamlined by clearly separating what makes up the environment (observation space, action space, rewards) and what makes up our reinforcement learning algorithm and training routines.

## Usage

We provide different versions of the bomberman game as environments. All Gym observation spaces are directly and purely computed from the original `game_state` provided to allow our model to plug into the original game again easily for the competition. 

### BomberGym-v0

`BomberGym-v0` is the most basic and naive environment. We simply translate the `game_state` into a 2-dimensional array:

```python
>>> import gym
>>> from bombergym.scenarios import coin_heaven
>>> from bombergym.environments import register
>>> 
>>> register() # Register our custom environments to gym
>>> settings, agents = coin_heaven() # Play in coin heaven scenario
>>> 
>>> env = gym.make('BomberGym-v1', args=settings, agents=agents)
>>> env.reset()
array([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
       [-2,  0,  0,  0,  0,  0,  0,  0,  3,  0,  3,  0,  3,  0,  3,  3, -2],
       [-2,  3, -2,  0, -2,  0, -2,  0, -2,  0, -2,  3, -2,  0, -2,  0, -2],
       [-2,  0,  3,  0,  0,  0,  0,  3,  3,  3,  0,  0,  0,  0,  0,  0, -2],
       [-2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  3, -2,  3, -2],
       [-2,  3,  0,  0,  3,  0,  0,  3,  0,  0,  0,  3,  0,  0,  3,  0, -2],
       [-2,  3, -2,  0, -2,  0, -2,  3, -2,  3, -2,  0, -2,  3, -2,  3, -2],
       [-2,  0,  0,  0,  0,  0,  0,  3,  0,  0,  3,  0,  0,  0,  0,  0, -2],
       [-2,  3, -2,  0, -2,  3, -2,  0, -2,  3, -2,  3, -2,  3, -2,  0, -2],
       [-2,  3,  0,  0,  0,  0,  3,  0,  3,  0,  3,  0,  0,  0,  3,  0, -2],
       [-2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2],
       [-2,  3,  0,  3,  0,  3,  3,  0,  0,  0,  0,  0,  0,  3,  0,  0, -2],
       [-2,  3, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2,  0, -2],
       [-2,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2],
       [-2,  3, -2,  0, -2,  0, -2,  0, -2,  3, -2,  0, -2,  3, -2,  0, -2],
       [-2,  0,  3,  0,  0,  3,  3,  0,  0,  3,  0,  0,  0,  3,  0,  2, -2],
       [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])
```

The array is simply a representation of the field as we see in the guy, where the numbers have the following meanings:

```
EXPLOSION = -4
BOMB = -3
WALL = -2
ENEMY = -1
EMPTY = 0
CRATE = 1
SELF = 2
COIN = 3
```

So far, we have not achieved great results from this environment alone.

### BomberGym-v1

Work in Progress!
The idea is to have a much reduced observation space.
We have some preliminary promising results with BomberGym-v1.