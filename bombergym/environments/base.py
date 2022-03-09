from abc import abstractmethod
import logging
from collections import namedtuple
from datetime import datetime
from typing import List, Dict

import numpy as np

import bombergym.original.events as e
from bombergym.original.agents import Agent, SequentialAgentBackend
from bombergym.original.items import Coin, Explosion, Bomb
import bombergym.settings as s

import gym
from gym import spaces
from bombergym.render import render


WorldArgs = namedtuple("WorldArgs",
                       ["no_gui", "fps", "turn_based", "update_interval", "save_replay", "replay", "make_video", "continue_without_training", "log_dir", "save_stats", "match_name", "seed", "silence_errors", "scenario"])

"""
Base environment, staying as close as possible to original environment.py.
"""
class BombeRLeWorld(gym.Env):
    metadata = {'render.modes': ['human']}
    logger: logging.Logger
    
    running: bool = False
    step_counter: int
    replay: Dict
    round_statistics: Dict
    
    agents: List[Agent]
    active_agents: List[Agent]
    arena: np.ndarray
    coins: List[Coin]
    bombs: List[Bomb]
    explosions: List[Explosion]
    user_input = 'WAIT'
    
    round_id: str

    def __init__(self, args: WorldArgs, agents):
      super().__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      # Needs to be overridden for now
      self.action_space = spaces.Discrete(len(s.ACTIONS))
      self.observation_space = spaces.Dict({})
      self.args = args
      self.setup_logging()

      self.colors = list(s.AGENT_COLORS)

      self.round = 0
      self.round_statistics = {}

      self.rng = np.random.default_rng(args.seed)

      self.running = False

      self.setup_agents(agents)
      self.last_state = None

    @abstractmethod
    def compute_extra_events(self, old_state: dict, new_state: dict, action):
        pass
    
    def reset(self):
        """Gym API reset"""
        raise NotImplementedError("Pls implement in specific BomberGym-vX environment!")

    def step(self, action):
        raise NotImplementedError("Pls implement in specific BomberGym-vX environment!")

    def render(self, mode="human", **kwargs):
        orig_state = self.get_state_for_agent(self.agents[0])
        return render(orig_state, **kwargs)

    def close(self):
        pass

    def setup_logging(self):
        self.logger = logging.getLogger('BombeRLeWorld')
        self.logger.setLevel(s.LOG_GAME)
        handler = logging.FileHandler(f'{self.args.log_dir}/game.log', mode="w")
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Initializing game world')

    def new_round(self):
        if self.running:
            self.logger.warning('New round requested while still running')
            self.end_round()

        new_round = self.round + 1
        self.logger.info(f'STARTING ROUND #{new_round}')

        # Bookkeeping
        self.step_counter = 0
        self.bombs = []
        self.explosions = []

        if self.args.match_name is not None:
            match_prefix = f"{self.args.match_name} | "
        else:
            match_prefix = ""
        self.round_id = f'{match_prefix}Round {new_round:02d} ({datetime.now().strftime("%Y-%m-%d %H-%M-%S")})'

        # Arena with wall and crate layout
        self.arena, self.coins, self.active_agents = self.build_arena()

        for agent in self.active_agents:
            agent.start_round()

        self.round = new_round
        self.running = True

    def add_agent(self, agent_dir, name, train=False):
        assert len(self.agents) < s.MAX_AGENTS

        # if self.args.single_process:
        backend = SequentialAgentBackend(train, name, agent_dir)
        # else:
        # backend = ProcessAgentBackend(train, name, agent_dir)
        backend.start()

        color = self.colors.pop()
        agent = Agent(name, agent_dir, name, train, backend, color, color)
        self.agents.append(agent)

    def tile_is_free(self, x, y):
        is_free = (self.arena[x, y] == 0)
        if is_free:
            for obstacle in self.bombs + self.active_agents:
                is_free = is_free and (obstacle.x != x or obstacle.y != y)
        return is_free

    def perform_agent_action(self, agent: Agent, action: str):
        # Perform the specified action if possible, wait otherwise
        if action == 'UP' and self.tile_is_free(agent.x, agent.y - 1):
            agent.y -= 1
            agent.add_event(e.MOVED_UP)
        elif action == 'DOWN' and self.tile_is_free(agent.x, agent.y + 1):
            agent.y += 1
            agent.add_event(e.MOVED_DOWN)
        elif action == 'LEFT' and self.tile_is_free(agent.x - 1, agent.y):
            agent.x -= 1
            agent.add_event(e.MOVED_LEFT)
        elif action == 'RIGHT' and self.tile_is_free(agent.x + 1, agent.y):
            agent.x += 1
            agent.add_event(e.MOVED_RIGHT)
        elif action == 'BOMB' and agent.bombs_left:
            self.logger.info(f'Agent <{agent.name}> drops bomb at {(agent.x, agent.y)}')
            self.bombs.append(Bomb((agent.x, agent.y), agent, s.BOMB_TIMER, s.BOMB_POWER))
            agent.bombs_left = False
            agent.add_event(e.BOMB_DROPPED)
        elif action == 'WAIT':
            agent.add_event(e.WAITED)
        else:
            agent.add_event(e.INVALID_ACTION)

    def do_step(self, user_input='WAIT'):
        assert self.running

        self.step_counter += 1
        self.logger.info(f'STARTING STEP {self.step_counter}')

        self.user_input = user_input
        self.logger.debug(f'User input: {self.user_input}')

        self.poll_and_run_agents()

        # Progress world elements based
        self.collect_coins()
        self.update_explosions()
        self.update_bombs()
        self.evaluate_explosions()
        events = self.active_agents[0].events if len(self.active_agents) else self.agents[0].events
        self.send_game_events()

        if self.time_to_stop():
            self.end_round()
        return events

    def collect_coins(self):
        for coin in self.coins:
            if coin.collectable:
                for a in self.active_agents:
                    if a.x == coin.x and a.y == coin.y:
                        coin.collectable = False
                        self.logger.info(f'Agent <{a.name}> picked up coin at {(a.x, a.y)} and receives 1 point')
                        a.update_score(s.REWARD_COIN)
                        a.add_event(e.COIN_COLLECTED)
                        #a.trophies.append(Trophy.coin_trophy)

    def update_explosions(self):
        # Progress explosions
        remaining_explosions = []
        for explosion in self.explosions:
            explosion.timer -= 1
            if explosion.timer <= 0:
                explosion.next_stage()
                if explosion.stage == 1:
                    explosion.owner.bombs_left = True
            if explosion.stage is not None:
                remaining_explosions.append(explosion)
        self.explosions = remaining_explosions

    def update_bombs(self):
        """
        Count down bombs placed
        Explode bombs at zero timer.

        :return:
        """
        for bomb in self.bombs:
            if bomb.timer <= 0:
                # Explode when timer is finished
                self.logger.info(f'Agent <{bomb.owner.name}>\'s bomb at {(bomb.x, bomb.y)} explodes')
                bomb.owner.add_event(e.BOMB_EXPLODED)
                blast_coords = bomb.get_blast_coords(self.arena)

                # Clear crates
                for (x, y) in blast_coords:
                    if self.arena[x, y] == 1:
                        self.arena[x, y] = 0
                        bomb.owner.add_event(e.CRATE_DESTROYED)
                        # Maybe reveal a coin
                        for c in self.coins:
                            if (c.x, c.y) == (x, y):
                                c.collectable = True
                                self.logger.info(f'Coin found at {(x, y)}')
                                bomb.owner.add_event(e.COIN_FOUND)

                # Create explosion
                screen_coords = [(s.GRID_OFFSET[0] + s.GRID_SIZE * x, s.GRID_OFFSET[1] + s.GRID_SIZE * y) for (x, y) in
                                 blast_coords]
                self.explosions.append(Explosion(blast_coords, screen_coords, bomb.owner, s.EXPLOSION_TIMER))
                bomb.active = False
            else:
                # Progress countdown
                bomb.timer -= 1
        self.bombs = [b for b in self.bombs if b.active]

    def evaluate_explosions(self):
        # Explosions
        agents_hit = set()
        for explosion in self.explosions:
            # Kill agents
            if explosion.is_dangerous():
                for a in self.active_agents:
                    if (not a.dead) and (a.x, a.y) in explosion.blast_coords:
                        agents_hit.add(a)
                        # Note who killed whom, adjust scores
                        if a is explosion.owner:
                            self.logger.info(f'Agent <{a.name}> blown up by own bomb')
                            a.add_event(e.KILLED_SELF)
                            #explosion.owner.trophies.append(Trophy.suicide_trophy)
                        else:
                            self.logger.info(f'Agent <{a.name}> blown up by agent <{explosion.owner.name}>\'s bomb')
                            self.logger.info(f'Agent <{explosion.owner.name}> receives 1 point')
                            explosion.owner.update_score(s.REWARD_KILL)
                            explosion.owner.add_event(e.KILLED_OPPONENT)
                            #explosion.owner.trophies.append(pygame.transform.smoothscale(a.avatar, (15, 15)))

        # Remove hit agents
        for a in agents_hit:
            a.dead = True
            self.active_agents.remove(a)
            a.add_event(e.GOT_KILLED)
            for aa in self.active_agents:
                if aa is not a:
                    aa.add_event(e.OPPONENT_ELIMINATED)

    def time_to_stop(self):
        # Check round stopping criteria
        if len(self.active_agents) == 0:
            self.logger.info(f'No agent left alive, wrap up round')
            return True

        if (len(self.active_agents) == 1
                and (self.arena == 1).sum() == 0
                and all([not c.collectable for c in self.coins])
                and len(self.bombs) + len(self.explosions) == 0):
            self.logger.info(f'One agent left alive with nothing to do, wrap up round')
            return True

        if any(a.train for a in self.agents) and not self.args.continue_without_training:
            if not any([a.train for a in self.active_agents]):
                self.logger.info('No training agent left alive, wrap up round')
                return True

        if self.step_counter >= s.MAX_STEPS:
            self.logger.info('Maximum number of steps reached, wrap up round')
            return True

        return False

    def setup_agents(self, agents):
        # Add specified agents and start their subprocesses
        self.agents = []
        for agent_dir, train in agents:
            if list([d for d, t in agents]).count(agent_dir) > 1:
                name = agent_dir + '_' + str(list([a.code_name for a in self.agents]).count(agent_dir))
            else:
                name = agent_dir
            self.add_agent(agent_dir, name, train=train)

    def build_arena(self):
        WALL = -1
        FREE = 0
        CRATE = 1
        arena = np.zeros((s.COLS, s.ROWS), int)

        scenario_info = s.SCENARIOS[self.args.scenario]

        # Crates in random locations
        arena[self.rng.random((s.COLS, s.ROWS)) < scenario_info["CRATE_DENSITY"]] = CRATE

        # Walls
        arena[:1, :] = WALL
        arena[-1:, :] = WALL
        arena[:, :1] = WALL
        arena[:, -1:] = WALL
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = WALL

        # Clean the start positions
        start_positions = [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]
        for (x, y) in start_positions:
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == 1:
                    arena[xx, yy] = FREE

        # Place coins at random, at preference under crates
        coins = []
        all_positions = np.stack(np.meshgrid(np.arange(s.COLS), np.arange(s.ROWS), indexing="ij"), -1)
        crate_positions = self.rng.permutation(all_positions[arena == CRATE])
        free_positions = self.rng.permutation(all_positions[arena == FREE])
        coin_positions = np.concatenate([
            crate_positions,
            free_positions
        ], 0)[:scenario_info["COIN_COUNT"]]
        for x, y in coin_positions:
            coins.append(Coin((x, y), collectable=arena[x, y] == FREE))

        # Reset agents and distribute starting positions
        active_agents = []
        for agent, start_position in zip(self.agents, self.rng.permutation(start_positions)):
            active_agents.append(agent)
            agent.x, agent.y = start_position

        return arena, coins, active_agents

    def get_state_for_agent(self, agent: Agent):
        if agent.dead:
            return None

        state = {
            'round': self.round,
            'step': self.step_counter,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': self.user_input,
        }

        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            if exp.is_dangerous():
                for (x, y) in exp.blast_coords:
                    explosion_map[x, y] = max(explosion_map[x, y], exp.timer - 1)
        state['explosion_map'] = explosion_map

        return state

    def poll_and_run_agents(self):
        # Tell agents to act
        # Do not run our own agent
        for i in range(1, len(self.active_agents)):
            a = self.active_agents[i]
            if a.available_think_time > 0:
                a.act(self.get_state_for_agent(a))

        for i in range(1, len(self.active_agents)):
            a = self.active_agents[i]
            action, _ = a.wait_for_act()
            self.perform_agent_action(a, action)

    def send_game_events(self):
        # Send events to all agents that expect them, then reset and wait for them
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.process_game_events(self.get_state_for_agent(a))
        for a in self.agents:
            if a.train:
                if not a.dead:
                    a.wait_for_game_event_processing()
        for a in self.active_agents:
            a.store_game_state(self.get_state_for_agent(a))
            a.reset_game_events()

    def end_round(self):
        if not self.running:
            raise ValueError('End-of-round requested while no round was running')
        # Wait in case there is still a game step running
        self.running = False

        for a in self.agents:
            a.note_stat("score", a.score)
            a.note_stat("rounds")
        self.round_statistics[self.round_id] = {
            "steps": self.step_counter,
            **{key: sum(a.statistics[key] for a in self.agents) for key in ["coins", "kills", "suicides"]}
        }

        self.logger.info(f'WRAPPING UP ROUND #{self.round}')
        # Clean up survivors
        for a in self.active_agents:
            a.add_event(e.SURVIVED_ROUND)

        # Send final event to agents that expect them
        for a in self.agents:
            if a.train:
                a.round_ended()

    def end(self):
        if self.running:
            self.end_round()