import gym
import getch
from stable_baselines3.common.env_checker import check_env
import bombergym.settings as s

# Way to avoid tedious argparse
class BombeRLeSettings:
    command_name = 'play'
    my_agent = None
    agents = [
        'gym_surrogate_agent', # Important, needs to be first. Represents our agent
        #'random_agent', # Possibility to add other agents here
    ]
    train = 1
    continue_without_training = False
    #scenario = 'coin-heaven'
    scenario = 'classic'
    seed = None
    n_rounds = 10 # Has no effect
    save_replay = False # Has no effect
    match_name = None # ?
    silence_errors = False
    skip_frames = False # No effect
    no_gui = False # No effect
    turn_based = False # No effect
    update_interval = .1 # No effect
    log_dir = './logs'
    save_stats = False # No effect ?
    make_video = False # No effect

bomber = BombeRLeSettings()

# Setup agents
agents = []
if bomber.train == 0 and not bomber.continue_without_training:
    bomber.continue_without_training = True
if bomber.my_agent:
    agents.append((bomber.my_agent, len(agents) < bomber.train))
    bomber.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
for agent_name in bomber.agents:
    agents.append((agent_name, len(agents) < bomber.train))

gym.envs.register(
    id='BomberGym-v0',
    entry_point='bombergym.environments.features:BombeRLeWorldFeatureEng',
    max_episode_steps=401,
    kwargs={ 'args': bomber, 'agents': agents }
)

env = gym.make('BomberGym-v0')
check_env(env)

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
    
 
