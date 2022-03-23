# Helper so we start always in the top left
def detect_starting_configuration(initial_obs):
    agent_frame = initial_obs[4, : , :]
