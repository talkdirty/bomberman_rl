import gym
import os

from argparse import ArgumentParser
import settings as s
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


parser = ArgumentParser()

subparsers = parser.add_subparsers(dest='command_name', required=True)

# Run arguments
play_parser = subparsers.add_parser("play")
agent_group = play_parser.add_mutually_exclusive_group()
agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                            help="First … agents should be set to training mode")
play_parser.add_argument("--continue-without-training", default=False, action="store_true")
# play_parser.add_argument("--single-process", default=False, action="store_true")

play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
play_parser.add_argument("--match-name", help="Give the match a name")

play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

group = play_parser.add_mutually_exclusive_group()
group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

# Replay arguments
replay_parser = subparsers.add_parser("replay")
replay_parser.add_argument("replay", help="File to load replay from")

# Interaction
for sub in [play_parser, replay_parser]:
    sub.add_argument("--turn-based", default=False, action="store_true",
                        help="Wait for key press until next movement")
    sub.add_argument("--update-interval", type=float, default=0.1,
                        help="How often agents take steps (ignored without GUI)")
    sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
    sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

    # Video?
    sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                        help="Make a video from the game")

args = parser.parse_args()

# Initialize environment and agents
if args.command_name == "play":
    agents = []
    if args.train == 0 and not args.continue_without_training:
        args.continue_without_training = True
    if args.my_agent:
        agents.append((args.my_agent, len(agents) < args.train))
        args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
    for agent_name in args.agents:
        agents.append((agent_name, len(agents) < args.train))

    gym.envs.register(
        id='BomberGym-v0',
        entry_point='bombergymenv:BombeRLeWorld',
        max_episode_steps=401,
        kwargs={ 'args': args, 'agents': agents }
    )
    #env = gym.make('BomberGym-v0')
    #check_env(env)

    env = make_vec_env("BomberGym-v0", n_envs=4)

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("bombermodel")