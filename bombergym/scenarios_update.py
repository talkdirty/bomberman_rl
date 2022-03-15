"""
Provide some basic Game scenarios. Modelled after the original recommended
scenarios from the Project sheet.
"""

class BasicSettings:
    """
    This is a remodelling of original game engines main.py argparse namespace.
    """
    command_name = 'play'
    my_agent = None
    agents = [
        'gym_surrogate_agent', # Important, needs to be first. Represents our agent
    ]
    train = 1
    continue_without_training = False
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

class MurderSettings(BasicSettings):
    scenario = 'murder'
    


class MurderEnemiesSettings(BasicSettings):
    scenario = 'murder'

    agents = [
        'gym_surrogate_agent', 
        'peaceful_agent',
        'peaceful_agent',
        'peaceful_agent',
    ]


def get_agents(settings):
    agents = []
    if settings.train == 0 and not settings.continue_without_training:
        settings.continue_without_training = True
    if settings.my_agent:
        agents.append((settings.my_agent, len(agents) < settings.train))
        settings.agents = ["peaceful_agent"] * (s.MAX_AGENTS - 1)
    for agent_name in settings.agents:
        agents.append((agent_name, len(agents) < settings.train))
    return agents


def murder():
    """
    agent plus peaceful agents. needs to blow them up.
    """
    settings = MurderEnemiesSettings()
    agents = get_agents(settings)

    return settings, agents



class CoinHeavenSettings(BasicSettings):
    scenario = 'coin-heaven'

class ClassicSettings(BasicSettings):
    scenario = 'classic'

class ClassicSettingsEnemies(BasicSettings):
    scenario = 'classic'
    agents = [
        'gym_surrogate_agent', 
        'rule_based_agent',
        'rule_based_agent',
        'rule_based_agent',
    ]

def get_agents(settings):
    agents = []
    if settings.train == 0 and not settings.continue_without_training:
        settings.continue_without_training = True
    if settings.my_agent:
        agents.append((settings.my_agent, len(agents) < settings.train))
        settings.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
    for agent_name in settings.agents:
        agents.append((agent_name, len(agents) < settings.train))
    return agents


def coin_heaven():
    """
    Basic coin heaven scenario. Only the agent. No enemies. No crates.
    """
    settings = CoinHeavenSettings()
    agents = get_agents(settings)
    return settings, agents

def classic():
    """
    Classic scenario. Only the agent. No enemies.
    """
    settings = ClassicSettings()
    agents = get_agents(settings)
    return settings, agents

def classic_with_opponents():
    """
    Classic scenario. 3 other agents.
    """
    settings = ClassicSettingsEnemies()
    agents = get_agents(settings)
    return settings, agents