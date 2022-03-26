import gym
import getch
from stable_baselines3.common.env_checker import check_env
from bombergym.scenarios import classic, classic_with_opponents
from bombergym.environments import register

register()
settings, agents = classic_with_opponents()

env = gym.make('BomberGym-v5', args=settings, agents=agents)
#check_env(env)

# env = make_vec_env("BomberGym-v0", n_envs=4)
env.reset()
env.render()
while True:
    inp = getch.getch()
    if inp == 'h':
        action = 3
    elif inp == 'j':
        action = 2
    elif inp == 'k':
        action = 0
    elif inp == 'l':
        action = 1
    elif inp == 'b':
        action = 5
    else:
        action = 4
    obs, rew, done, other = env.step(action)
    if not done:
        # temp?
        feature_info = other["features"] if "features" in other else None
        env.render(events=other["events"], rewards=rew, other=feature_info)
    else:
        print(other["events"], f"Reward: {rew}")
        break
    
 
