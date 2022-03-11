import gym 

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