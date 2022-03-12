import gym 

import bombergym.environments.manhattan_v2.features

def register():
    """
    Registers all custom environments to gym
    """
    gym.envs.register(
        id='BomberGym-v0',
        entry_point=f'{__package__}.plain.plain:BomberGymPlain',
        max_episode_steps=401,
    )

    gym.envs.register(
        id='BomberGym-v1',
        entry_point=f'{__package__}.reduced.reduced:BomberGymReduced',
        max_episode_steps=401,
    )

    gym.envs.register(
        id='BomberGym-v2',
        entry_point=f'{__package__}.manhattan.manhattan:BomberGymReducedManhattanNorm',
        max_episode_steps=401,
    )

    gym.envs.register(
        id='BomberGym-v3',
        entry_point=f'{__package__}.manhattan.manhattan:BomberGymReducedManhattanNorm',
        max_episode_steps=401,
        kwargs={'state_fn': bombergym.environments.manhattan_v2.features.state_to_gym}
    )